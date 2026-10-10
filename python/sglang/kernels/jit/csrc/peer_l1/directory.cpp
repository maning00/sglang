#include "directory.h"

#include <exception>
#include <string>

using sglang::peer_l1::Directory;
using sglang::peer_l1::Hash;

namespace {
thread_local std::string last_error;
template <class F>
int64_t checked(F&& fn) noexcept {
  try {
    return fn();
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}
}  // namespace

extern "C" {
const char* l1_error() {
  return last_error.c_str();
}
int64_t l1_bytes(uint32_t rows) {
  return checked([&] { return Directory::bytes(rows); });
}
int64_t l1_init(void* p, uint32_t rows, uint64_t cap) {
  return checked([&] {
    Directory::initialize(p, rows, cap);
    return 0;
  });
}
int64_t l1_publish(void* p, const Hash* hashes, const int64_t* rows, size_t n) {
  return checked([&] {
    Directory d(p);
    for (size_t i = 0; i < n; ++i)
      d.publish(rows[i], hashes[i]);
    return 0;
  });
}
int64_t l1_retire(void* p, const Hash* hashes, int64_t* rows, size_t n) {
  return checked([&] {
    Directory d(p);
    for (size_t i = 0; i < n; ++i)
      rows[i] = d.retire(hashes[i]);
    return 0;
  });
}
int64_t l1_probe(void* p, const Hash* hashes, int64_t* rows, int64_t* gens, size_t n) {
  return checked([&] {
    Directory d(p);
    size_t i = 0;
    for (; i < n && d.probe(hashes[i], rows[i], gens[i]); ++i) {
    }
    return i;
  });
}
int64_t l1_pin(void* p, const int64_t* rows, const int64_t* gens, size_t n) {
  return checked([&] { return Directory(p).pin(rows, gens, n); });
}
int64_t l1_unpin(void* p, const int64_t* rows, size_t n) {
  return checked([&] {
    Directory(p).unpin(rows, n);
    return 0;
  });
}
int64_t l1_states(void* p, const int64_t* rows, int64_t* pins, int64_t* live, size_t n) {
  return checked([&] {
    Directory d(p);
    for (size_t i = 0; i < n; ++i) {
      pins[i] = d.pins(rows[i]);
      live[i] = d.live(rows[i]);
    }
    return 0;
  });
}
int64_t l1_refs(void* p) {
  return checked([&] { return Directory(p).pinned_refs(); });
}
int64_t l1_freeze(void* p) {
  return checked([&] { return Directory(p).freeze(); });
}
int64_t l1_thaw(void* p) {
  return checked([&] {
    Directory(p).thaw();
    return 0;
  });
}
int64_t l1_rebuild(void* p) {
  return checked([&] {
    Directory(p).rebuild();
    return 0;
  });
}
}
