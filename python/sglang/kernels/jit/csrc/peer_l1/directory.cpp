#include "directory.h"

#include <exception>
#include <string>

using sglang::peer_l1::Directory;

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
}
