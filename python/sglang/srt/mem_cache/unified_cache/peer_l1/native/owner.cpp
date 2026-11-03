#include "owner.h"

#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
template <class F> int64_t checked(F &&fn) noexcept {
  try {
    return fn();
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
} // namespace

extern "C" const char *l1_owner_error() { return last_error.c_str(); }

extern "C" {
int64_t l1_owner_start(void *memory, const char *endpoint, uint64_t epoch,
                       uint64_t cap, int lease_ms) {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new sglang::peer_l1::Owner(memory, endpoint, epoch, cap, lease_ms));
  });
}
const char *l1_owner_endpoint(void *owner) {
  return static_cast<sglang::peer_l1::Owner *>(owner)->endpoint().c_str();
}
int64_t l1_owner_listen_tcp(void *owner, const char *endpoint) {
  return checked([&] {
    static_cast<sglang::peer_l1::Owner *>(owner)->listen_tcp(endpoint);
    return int64_t(0);
  });
}
const char *l1_owner_tcp_endpoint(void *owner) {
  return static_cast<sglang::peer_l1::Owner *>(owner)->tcp_endpoint().c_str();
}
int64_t l1_owner_stop(void *owner) {
  delete static_cast<sglang::peer_l1::Owner *>(owner);
  return 0;
}
int64_t l1_owner_describe(void *owner, const char *data, size_t size) {
  return checked([&] {
    static_cast<sglang::peer_l1::Owner *>(owner)->describe(data, size);
    return int64_t(0);
  });
}
int64_t l1_owner_publish(void *owner, const sglang::peer_l1::Key *keys,
                         const int64_t *rows, size_t n) {
  return checked([&] {
    static_cast<sglang::peer_l1::Owner *>(owner)->publish(keys, rows, n);
    return 0;
  });
}
int64_t l1_owner_retire(void *owner, const sglang::peer_l1::Key *keys,
                        int64_t *rows, size_t n) {
  return checked([&] {
    static_cast<sglang::peer_l1::Owner *>(owner)->retire(keys, rows, n);
    return 0;
  });
}
int64_t l1_owner_rpc(const char *endpoint, uint64_t op, uint64_t epoch,
                     uint64_t reader, uint64_t seq,
                     const sglang::peer_l1::Key *keys, int64_t *out, size_t n,
                     int timeout_ms) {
  sglang::peer_l1::Request request;
  request.op = op;
  request.epoch = epoch;
  request.reader = reader;
  request.seq = seq;
  request.n = n;
  return sglang::peer_l1::rpc(endpoint, request, keys, out, timeout_ms);
}
int64_t l1_cleanup_start(int timeout_ms) {
  return checked([&] {
    return reinterpret_cast<int64_t>(
        new sglang::peer_l1::CleanupQueue(timeout_ms));
  });
}
int64_t l1_cleanup_stop(void *queue) {
  delete static_cast<sglang::peer_l1::CleanupQueue *>(queue);
  return 0;
}
int64_t l1_cleanup_pending(void *queue) {
  return static_cast<sglang::peer_l1::CleanupQueue *>(queue)->pending();
}
int64_t l1_cleanup_create(void *queue, const char *endpoint, uint64_t op,
                          uint64_t epoch, uint64_t reader, uint64_t seq,
                          uint64_t lease_ms) {
  return checked([&] {
    sglang::peer_l1::Request request;
    request.op = op;
    request.epoch = epoch;
    request.reader = reader;
    request.seq = seq;
    request.n = 0;
    auto *item = new sglang::peer_l1::Cleanup{
        static_cast<sglang::peer_l1::CleanupQueue *>(queue), endpoint, request};
    if (lease_ms)
      item->expires =
          sglang::peer_l1::Clock::now() + std::chrono::milliseconds(lease_ms);
    return reinterpret_cast<int64_t>(item);
  });
}
void l1_cleanup_enqueue(void *item) { sglang::peer_l1::completed_copy(item); }
}
