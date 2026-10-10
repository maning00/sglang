// Single-writer, multi-reader directory over process-shared, cache-coherent RAM.
// Readers pin a physical row and generation, never a tree node.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

namespace sglang::peer_l1 {

static_assert(std::atomic<uint64_t>::is_always_lock_free);
constexpr uint64_t kMagic = 0x504545524c310001ULL;
constexpr uint64_t kFrozen = uint64_t{1} << 63;
constexpr uint64_t kLive = 1;
constexpr uint64_t kPin = 2;
constexpr uint64_t kGeneration = uint64_t{1} << 32;
constexpr uint64_t kTombstone = ~uint64_t{0};

struct Hash {
  uint64_t lo, hi;
};

struct alignas(64) Header {
  uint64_t magic;
  uint32_t rows, capacity;
  uint64_t pin_cap;
  std::atomic<uint64_t> pin_word;
};

struct alignas(32) Row {
  std::atomic<uint64_t> state;
  // Atomic metadata avoids a C++ data race with speculative probes of retired rows.
  std::atomic<uint64_t> lo, hi;
};

class Directory {
 public:
  static uint32_t capacity(uint32_t rows) {
    if (!rows || rows > (1u << 28)) throw std::invalid_argument("invalid peer L1 row count");
    uint32_t c = 1;
    while (c < 2 * rows)
      c <<= 1;
    return c;
  }

  static size_t bytes(uint32_t rows) {
    return sizeof(Header) + sizeof(Row) * rows + sizeof(std::atomic<uint64_t>) * capacity(rows);
  }

  static void initialize(void* memory, uint32_t rows, uint64_t pin_cap) {
    if (!pin_cap || pin_cap >= kFrozen) throw std::invalid_argument("invalid peer L1 pin cap");
    auto* h = new (memory) Header{};
    h->rows = rows;
    h->capacity = capacity(rows);
    h->pin_cap = pin_cap;
    auto* r = reinterpret_cast<Row*>(h + 1);
    for (uint32_t i = 0; i < rows; ++i)
      new (r + i) Row{};
    auto* table = reinterpret_cast<std::atomic<uint64_t>*>(r + rows);
    for (uint32_t i = 0; i < h->capacity; ++i)
      new (table + i) std::atomic<uint64_t>{0};
    h->magic = kMagic;  // Exchanged with peers only after initialization completes.
  }

  explicit Directory(void* memory) : h_(static_cast<Header*>(memory)), rows_(reinterpret_cast<Row*>(h_ + 1)) {
    if (h_->magic != kMagic || h_->capacity != capacity(h_->rows))
      throw std::invalid_argument("incompatible peer L1 directory");
    table_ = reinterpret_cast<std::atomic<uint64_t>*>(rows_ + h_->rows);
  }

  uint64_t pins(uint32_t row) const {
    return (checked(row).state.load(std::memory_order_acquire) >> 1) & 0x7fffffff;
  }
  uint64_t pinned_refs() const {
    return h_->pin_word.load(std::memory_order_acquire) & ~kFrozen;
  }
  bool live(uint32_t row) const {
    return checked(row).state.load(std::memory_order_acquire) & kLive;
  }
  bool freeze() {
    uint64_t expected = 0;
    return h_->pin_word.compare_exchange_strong(expected, kFrozen, std::memory_order_acq_rel);
  }
  void thaw() {
    h_->pin_word.fetch_and(~kFrozen, std::memory_order_release);
  }

  bool probe(Hash hash, int64_t& row, int64_t& gen) const {
    const auto index = find(hash);
    if (index == h_->capacity) return false;
    const auto entry = table_[index].load(std::memory_order_acquire);
    if (!matches(entry, hash)) return false;
    row = static_cast<uint32_t>(entry) - 1;
    gen = entry >> 32;
    return true;
  }

  bool pin(const int64_t* rows, const int64_t* gens, size_t count) {
    if (!count) return true;
    auto word = h_->pin_word.load(std::memory_order_acquire);
    do {
      if ((word & kFrozen) || count > h_->pin_cap || word > h_->pin_cap - count) return false;
    } while (!h_->pin_word.compare_exchange_weak(word, word + count, std::memory_order_acq_rel));
    size_t done = 0;
    for (; done < count; ++done) {
      if (rows[done] < 0 || static_cast<uint64_t>(rows[done]) >= h_->rows) break;
      auto& state = rows_[rows[done]].state;
      auto value = state.load(std::memory_order_acquire);
      bool acquired = false;
      while ((value & kLive) && (value >> 32) == static_cast<uint64_t>(gens[done]) &&
             ((value >> 1) & 0x7fffffff) != 0x7fffffff) {
        if (state.compare_exchange_weak(value, value + kPin, std::memory_order_acq_rel)) {
          acquired = true;
          break;
        }
      }
      if (!acquired) break;
    }
    if (done == count) return true;
    for (size_t i = 0; i < done; ++i)
      rows_[rows[i]].state.fetch_sub(kPin, std::memory_order_release);
    h_->pin_word.fetch_sub(count, std::memory_order_release);
    return false;
  }

  // Called by the copy stream's native host callback. The reservation owns the
  // exact rows and directory mapping until this operation completes.
  void unpin(const int64_t* rows, size_t count) {
    for (size_t i = 0; i < count; ++i)
      rows_[rows[i]].state.fetch_sub(kPin, std::memory_order_release);
    h_->pin_word.fetch_sub(count, std::memory_order_release);
  }

  void publish(uint32_t row, Hash hash) {
    auto& r = checked(row);
    const auto state = r.state.load(std::memory_order_acquire);
    if (state & 0xffffffffULL) throw std::logic_error("publishing a live or pinned peer L1 row");
    if (h_->pin_word.load(std::memory_order_acquire) & kFrozen)
      throw std::logic_error("publishing a frozen peer L1 directory");
    if (find(hash) != h_->capacity) throw std::logic_error("peer L1 hash must be retired before replacement");
    r.lo.store(hash.lo, std::memory_order_relaxed);
    r.hi.store(hash.hi, std::memory_order_relaxed);
    r.state.store(state | kLive, std::memory_order_release);
    insert(row, state >> 32, hash);
  }

  // Returns the physical row, or -1 for a not-yet-published attach. The owner
  // separately cancels pending publications before freeing their slots.
  int64_t retire(Hash hash) {
    const auto index = find(hash);
    if (index == h_->capacity) return -1;
    const auto entry = table_[index].load(std::memory_order_relaxed);
    const auto row = static_cast<uint32_t>(entry) - 1;
    auto& state = rows_[row].state;
    auto value = state.load(std::memory_order_acquire);
    do {
      if ((value >> 32) == 0xfffffffe) throw std::overflow_error("peer L1 generation exhausted");
    } while (!state.compare_exchange_weak(value, (value & ~kLive) + kGeneration, std::memory_order_acq_rel));
    table_[index].store(kTombstone, std::memory_order_release);
    return row;
  }

  // The single owner can compact tombstones while readers probe. A concurrent
  // probe may miss, but a successful pin still validates the row generation.
  void rebuild() {
    for (uint32_t i = 0; i < h_->capacity; ++i)
      table_[i].store(0, std::memory_order_release);
    for (uint32_t i = 0; i < h_->rows; ++i) {
      const auto state = rows_[i].state.load(std::memory_order_acquire);
      if (state & kLive) insert(i, state >> 32, {rows_[i].lo.load(), rows_[i].hi.load()});
    }
  }

 private:
  Row& checked(uint32_t row) const {
    if (row >= h_->rows) throw std::out_of_range("peer L1 row exceeds pool");
    return rows_[row];
  }

  bool matches(uint64_t entry, Hash hash) const {
    if (!entry || entry == kTombstone) return false;
    const auto row = static_cast<uint32_t>(entry) - 1;
    if (row >= h_->rows) return false;
    const auto& r = rows_[row];
    const auto before = r.state.load(std::memory_order_acquire);
    if (!(before & kLive) || (before >> 32) != (entry >> 32)) return false;
    const bool equal =
        r.lo.load(std::memory_order_relaxed) == hash.lo && r.hi.load(std::memory_order_relaxed) == hash.hi;
    std::atomic_thread_fence(std::memory_order_acquire);
    // Pin-count changes do not change row identity. In particular, an owner
    // must never miss a retirement merely because a reader acquired a pin.
    return equal && ((r.state.load(std::memory_order_relaxed) ^ before) & 0xffffffff00000001ULL) == 0;
  }

  uint32_t find(Hash hash) const {
    auto index = static_cast<uint32_t>(hash.lo) & (h_->capacity - 1);
    for (uint32_t n = 0; n < h_->capacity; ++n, index = (index + 1) & (h_->capacity - 1)) {
      const auto entry = table_[index].load(std::memory_order_acquire);
      if (!entry) break;
      if (matches(entry, hash)) return index;
    }
    return h_->capacity;
  }

  void insert(uint32_t row, uint64_t gen, Hash hash) {
    auto index = static_cast<uint32_t>(hash.lo) & (h_->capacity - 1);
    for (uint32_t n = 0; n < h_->capacity; ++n, index = (index + 1) & (h_->capacity - 1)) {
      const auto old = table_[index].load(std::memory_order_relaxed);
      if (!old || old == kTombstone) {
        table_[index].store((gen << 32) | (static_cast<uint64_t>(row) + 1), std::memory_order_release);
        return;
      }
    }
    throw std::logic_error("peer L1 directory is full");
  }

  Header* h_;
  Row* rows_;
  std::atomic<uint64_t>* table_;
};

}  // namespace sglang::peer_l1
