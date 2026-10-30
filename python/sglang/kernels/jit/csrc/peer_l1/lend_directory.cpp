// A disposable node-local hint index. The decoder's generation check is authoritative.
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <pthread.h>

struct Value {
  int64_t rank, page, generation;
};
struct Entry {
  uint8_t key[32];
  Value value;
  uint8_t state;
};
struct Header {
  pthread_mutex_t mutex;
  uint64_t capacity;
};
static Entry* entries(Header* h) {
  return reinterpret_cast<Entry*>(h + 1);
}
struct Guard {
  Header* h;
  explicit Guard(Header* p) : h(p) {
    int rc = pthread_mutex_lock(&h->mutex);
    if (rc == EOWNERDEAD) pthread_mutex_consistent(&h->mutex);
  }
  ~Guard() {
    pthread_mutex_unlock(&h->mutex);
  }
};
static uint64_t hash_key(const uint8_t* key) {
  uint64_t hash = 1469598103934665603ULL;
  for (int i = 0; i < 32; ++i)
    hash = (hash ^ key[i]) * 1099511628211ULL;
  return hash;
}
static uint64_t slot(Header* h, const uint8_t* key, bool insert) {
  uint64_t hash = hash_key(key) % h->capacity;
  for (uint64_t n = 0; n < h->capacity; ++n) {
    uint64_t s = (hash + n) % h->capacity;
    Entry& e = entries(h)[s];
    if (e.state == 0) return insert ? s : h->capacity;
    if (e.state == 1 && !std::memcmp(e.key, key, 32)) return s;
  }
  return h->capacity;
}
static void erase(Header* h, uint64_t hole) {
  auto* table = entries(h);
  table[hole].state = 0;
  // Move entries whose probe chain crosses the hole. Deletions leave a truly
  // empty slot, so misses remain cheap after repeated fill/evict cycles.
  uint64_t s = hole;
  for (uint64_t n = 1; n < h->capacity; ++n) {
    s = (s + 1) % h->capacity;
    if (table[s].state == 0) break;
    uint64_t home = hash_key(table[s].key) % h->capacity;
    if ((hole + h->capacity - home) % h->capacity < (s + h->capacity - home) % h->capacity) {
      table[hole] = table[s];
      table[s].state = 0;
      hole = s;
    }
  }
}
extern "C" {
uint64_t lend_bytes(uint64_t capacity) {
  return sizeof(Header) + capacity * sizeof(Entry);
}
int lend_init(void* memory, uint64_t capacity) {
  auto* h = static_cast<Header*>(memory);
  std::memset(memory, 0, lend_bytes(capacity));
  h->capacity = capacity;
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
  int rc = pthread_mutex_init(&h->mutex, &attr);
  pthread_mutexattr_destroy(&attr);
  return rc;
}
void lend_query(void* memory, const uint8_t* keys, Value* values, uint64_t count) {
  auto* h = static_cast<Header*>(memory);
  Guard guard(h);
  for (uint64_t i = 0; i < count; ++i) {
    auto s = slot(h, keys + i * 32, false);
    values[i] = s < h->capacity ? entries(h)[s].value : Value{-1, 0, 0};
  }
}
int lend_put(void* memory, const uint8_t* keys, const Value* values, uint64_t count) {
  auto* h = static_cast<Header*>(memory);
  Guard guard(h);
  int written = 0;
  for (uint64_t i = 0; i < count; ++i) {
    auto s = slot(h, keys + i * 32, true);
    if (s == h->capacity) continue;
    auto& e = entries(h)[s];
    std::memcpy(e.key, keys + i * 32, 32);
    e.value = values[i];
    e.state = 1;
    ++written;
  }
  return written;
}
void lend_remove(void* memory, const uint8_t* keys, const Value* values, uint64_t count) {
  auto* h = static_cast<Header*>(memory);
  Guard guard(h);
  for (uint64_t i = 0; i < count; ++i) {
    auto s = slot(h, keys + i * 32, false);
    if (s == h->capacity) continue;
    auto& e = entries(h)[s];
    if (!std::memcmp(&e.value, &values[i], sizeof(Value))) erase(h, s);
  }
}
}
