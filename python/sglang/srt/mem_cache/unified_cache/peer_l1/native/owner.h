#pragma once
#include "rpc.h"
#include "tcp.h"
#include <sys/eventfd.h>
#include <sys/socket.h>

#include "directory.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace sglang::peer_l1 {
/// The authoritative directory lock also orders local retire and CONTAINS.
class Owner {
public:
  Owner(void *memory, const char *endpoint, uint64_t epoch, uint64_t cap,
        int lease_ms)
      : directory_(memory), epoch_(epoch), cap_(cap),
        row_count_(static_cast<Header *>(memory)->rows), lease_ms_(lease_ms),
        full_keys_(row_count_) {
    auto addr = address(endpoint);
    family_ = addr.storage.ss_family;
    if (family_ != AF_UNIX) {
      listen_tcp(endpoint);
      endpoint_ = tcp_->endpoint();
      return;
    }
    fd_ = socket(family_, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd_ < 0)
      throw std::runtime_error("peer L1 socket failed");
    int one = 1;
    if ((family_ != AF_UNIX &&
         setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))) ||
        bind(fd_, addr.data(), addr.length) || listen(fd_, 128)) {
      close(fd_);
      throw std::runtime_error("peer L1 bind failed");
    }
    try {
      endpoint_ = bound_endpoint(fd_, endpoint, addr);
      thread_ = std::thread([this] { run(); });
    } catch (...) {
      close(fd_);
      throw;
    }
  }
  const std::string &endpoint() const { return endpoint_; }
  const std::string &listen_tcp(const char *endpoint) {
    if (tcp_)
      throw std::runtime_error("peer L1 TCP listener already started");
    tcp_ = std::make_unique<TcpServer>(
        endpoint, epoch_,
        [this](const Request &request, const std::vector<Key> &keys) {
          return dispatch(request, keys);
        });
    return tcp_->endpoint();
  }
  const std::string &tcp_endpoint() const { return tcp_->endpoint(); }
  ~Owner() {
    tcp_.reset();
    stop_.store(true);
    if (thread_.joinable())
      thread_.join();
    if (fd_ >= 0)
      close(fd_);
    // Only engine teardown may destroy outstanding pins without RELEASE.
  }
  void publish(const Key *keys, const int64_t *rows, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
      directory_.publish(rows[i], {keys[i][0], keys[i][1]});
      full_keys_[rows[i]] = keys[i];
    }
  }
  void retire(const Key *keys, int64_t *rows, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
      int64_t row, gen;
      rows[i] = -1;
      if (find(keys[i], row, gen)) {
        rows[i] = directory_.retire({keys[i][0], keys[i][1]});
      }
    }
    retired_ += n;
    if (retired_ >= row_count_) {
      directory_.rebuild();
      retired_ = 0;
    }
  }

private:
  struct Grant {
    std::vector<int64_t> rows;
    Clock::time_point created;
  };
  using Identity = std::pair<uint64_t, uint64_t>;
  bool find(const Key &key, int64_t &row, int64_t &gen) {
    if (!directory_.probe({key[0], key[1]}, row, gen))
      return false;
    return full_keys_[row] == key;
  }
  void finish(uint64_t reader, uint64_t seq) {
    auto saved = grants_.find({reader, seq});
    if (saved != grants_.end()) {
      directory_.unpin(saved->second.rows.data(), saved->second.rows.size());
      grants_.erase(saved);
    }
    // A completion watermark also rejects cancelled requests arriving late.
    completed_[reader] = std::max(completed_[reader], seq);
  }
  std::vector<int64_t> dispatch(const Request &request,
                                const std::vector<Key> &keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    Identity id{request.reader, request.seq};
    auto saved = grants_.find(id);
    if (request.op == Cleanups) {
      for (const auto &key : keys)
        finish(key[0], key[1]);
      return {};
    }
    if (request.op == Release || request.op == Cancel) {
      finish(request.reader, request.seq);
      return {};
    }
    if (request.op == AcquireExact) {
      if (saved != grants_.end())
        return saved->second.rows;
      if (request.seq <= completed_[request.reader])
        return {};
      for (auto grant = grants_.lower_bound({request.reader, 0});
           grant != grants_.end() && grant->first.first == request.reader;
           ++grant) {
        if (Clock::now() - grant->second.created >
            std::chrono::milliseconds(lease_ms_))
          return {};
      }
    }
    std::vector<int64_t> rows, gens;
    auto limit = request.op == Contains ? keys.size()
                                        : std::min<uint64_t>(keys.size(), cap_);
    for (size_t i = 0; i < limit; ++i) {
      int64_t row, gen;
      bool found = find(keys[i], row, gen);
      if (request.op == Contains)
        rows.push_back(found);
      else {
        if (!found)
          break;
        rows.push_back(row);
        gens.push_back(gen);
      }
    }
    if (request.op == AcquireExact) {
      if (rows.size() != keys.size()) {
        completed_[request.reader] =
            std::max(completed_[request.reader], request.seq);
        return {};
      }
      // Allocate before pinning so a lost reply remains cancellable.
      grants_.emplace(id, Grant{rows, Clock::now()});
      if (rows.empty() ||
          !directory_.pin(rows.data(), gens.data(), rows.size())) {
        grants_.erase(id);
        completed_[request.reader] =
            std::max(completed_[request.reader], request.seq);
        return {};
      }
    }
    return rows;
  }
  void serve(int fd) {
    Request request{};
    auto deadline = Clock::now() + std::chrono::milliseconds(100);
    if (!transfer_bytes(fd, &request, sizeof(request), false, deadline) ||
        request.magic != Request{}.magic || request.n > (1 << 20) ||
        (request.op != Lookup && request.op != Release &&
         request.op != Cancel && request.op != Contains &&
         request.op != Cleanups && request.op != AcquireExact))
      return;
    std::vector<Key> keys(request.n);
    if (!transfer_bytes(fd, keys.data(), keys.size() * sizeof(Key), false,
                        deadline))
      return;
    std::vector<int64_t> result;
    if (request.epoch == epoch_)
      result = dispatch(request, keys);
    Reply reply{epoch_, result.size()};
    if (transfer_bytes(fd, &reply, sizeof(reply), true, deadline))
      transfer_bytes(fd, result.data(), result.size() * sizeof(int64_t), true,
                     deadline);
  }
  void run() noexcept {
    while (!stop_.load()) {
      pollfd descriptor{fd_, POLLIN, 0};
      if (poll(&descriptor, 1, 10) <= 0)
        continue;
      int fd = accept4(fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0)
        continue;
      try {
        if (no_delay(fd, family_))
          serve(fd);
      } catch (...) { /* Lost replies are resolved by reliable cancellation. */
      }
      close(fd);
    }
  }
  Directory directory_;
  uint64_t epoch_, cap_, row_count_, retired_ = 0;
  int lease_ms_, fd_ = -1, family_;
  std::unique_ptr<TcpServer> tcp_;
  std::string endpoint_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mutex_;
  std::vector<Key> full_keys_;
  std::map<Identity, Grant> grants_;
  std::map<uint64_t, uint64_t> completed_;
};

class CleanupQueue;
struct Cleanup {
  CleanupQueue *queue;
  std::string endpoint;
  Request request;
  Cleanup *next = nullptr;
  Clock::time_point retry{};
  int delay_ms = 1;
};

/// Host callbacks only publish preallocated nodes; the sender owns retries and
/// deletion.
class CleanupQueue {
public:
  explicit CleanupQueue(int timeout_ms)
      : timeout_ms_(timeout_ms), wake_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (wake_ < 0)
      throw std::runtime_error("peer L1 eventfd failed");
    thread_ = std::thread([this] { run(); });
  }
  ~CleanupQueue() {
    stop_.store(true);
    wake();
    thread_.join();
    close(wake_);
    auto *node = head_.exchange(nullptr);
    while (node) {
      auto *next = node->next;
      delete node;
      node = next;
    }
  }
  void enqueue(Cleanup *node) noexcept {
    pending_.fetch_add(1);
    auto *head = head_.load(std::memory_order_relaxed);
    do {
      node->next = head;
    } while (!head_.compare_exchange_weak(head, node, std::memory_order_release,
                                          std::memory_order_relaxed));
    wake();
  }
  uint64_t pending() const { return pending_.load(); }

private:
  void wake() noexcept {
    uint64_t one = 1;
    while (write(wake_, &one, sizeof(one)) < 0 && errno == EINTR) {
    }
  }
  void run() noexcept {
    std::vector<Cleanup *> pending;
    while (!stop_.load()) {
      auto *node = head_.exchange(nullptr, std::memory_order_acquire);
      while (node) {
        auto *next = node->next;
        pending.push_back(node);
        node = next;
      }
      for (size_t i = 0; i < pending.size(); ++i) {
        auto *first = pending[i];
        if (!first || Clock::now() < first->retry)
          continue;
        std::vector<size_t> positions;
        std::vector<Key> requests;
        for (size_t j = i; j < pending.size() && positions.size() < 64; ++j) {
          auto *item = pending[j];
          if (item && item->endpoint == first->endpoint &&
              item->request.epoch == first->request.epoch &&
              Clock::now() >= item->retry) {
            positions.push_back(j);
            requests.push_back({item->request.reader, item->request.seq, 0, 0});
          }
        }
        Request request = first->request;
        request.op = Cleanups;
        request.n = requests.size();
        auto result = rpc(first->endpoint.c_str(), request, requests.data(),
                          nullptr, timeout_ms_);
        for (auto position : positions) {
          auto *item = pending[position];
          if (result == 0 || result == -2) {
            delete item;
            pending[position] = nullptr;
            pending_.fetch_sub(1);
          } else {
            item->retry =
                Clock::now() + std::chrono::milliseconds(item->delay_ms);
            item->delay_ms = std::min(1000, item->delay_ms * 2);
          }
        }
      }
      pending.erase(std::remove(pending.begin(), pending.end(), nullptr),
                    pending.end());
      int timeout = -1;
      for (auto *item : pending) {
        auto left = std::max<int64_t>(
            0, std::chrono::duration_cast<std::chrono::milliseconds>(
                   item->retry - Clock::now())
                   .count());
        timeout = timeout < 0 ? left : std::min<int64_t>(timeout, left);
      }
      pollfd descriptor{wake_, POLLIN, 0};
      if (poll(&descriptor, 1, timeout) > 0) {
        uint64_t count;
        while (read(wake_, &count, sizeof(count)) < 0 && errno == EINTR) {
        }
      }
    }
    for (auto *node : pending)
      delete node;
  }
  int timeout_ms_, wake_;
  std::atomic<bool> stop_{false};
  std::atomic<Cleanup *> head_{nullptr};
  std::atomic<uint64_t> pending_{0};
  std::thread thread_;
};
static_assert(std::atomic<Cleanup *>::is_always_lock_free);
inline void completed_copy(void *context) noexcept {
  auto *item = static_cast<Cleanup *>(context);
  item->queue->enqueue(item);
}
} // namespace sglang::peer_l1
