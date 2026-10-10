// Standalone protocol stress: real process-shared atomics and mutable payloads.
// c++ -O3 -std=c++17 -Ipython/sglang/kernels/jit/csrc/peer_l1 \
//     test/manual/peer_l1/stress_directory.cpp -o /tmp/stress_peer_l1
#include "directory.h"

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

using sglang::peer_l1::Directory;
using sglang::peer_l1::Hash;
constexpr int kRows = 64;
constexpr int kReaders = 3;

struct Shared {
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> reads{0};
  std::atomic<uint64_t> data[kRows][8]{};
};

int main(int argc, char** argv) {
  const uint64_t rounds = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 5000000;
  void* memory = mmap(nullptr, Directory::bytes(kRows), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  auto* shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0));
  assert(memory != MAP_FAILED && shared != MAP_FAILED);
  new (shared) Shared;
  Directory::initialize(memory, kRows, kRows);
  Directory directory(memory);
  for (int row = 0; row < kRows; ++row) {
    for (int word = 0; word < 8; ++word) shared->data[row][word].store(word);
    directory.publish(row, Hash{static_cast<uint64_t>(row), 123});
  }
  pid_t readers[kReaders];
  for (int reader = 0; reader < kReaders; ++reader) {
    readers[reader] = fork();
    assert(readers[reader] >= 0);
    if (!readers[reader]) {
      uint64_t iteration = reader;
      uint64_t reads = 0;
      while (!shared->stop.load(std::memory_order_acquire)) {
        const auto candidate = (iteration++ * 17) % kRows;
        int64_t row, generation;
        if (!directory.probe(Hash{candidate, 123}, row, generation)) continue;
        if (!directory.pin(&row, &generation, 1)) continue;
        for (int word = 0; word < 8; ++word) {
          if (shared->data[row][word].load(std::memory_order_relaxed) != static_cast<uint64_t>(generation * 8 + word))
            std::_Exit(1);
        }
        directory.unpin(&row, 1);
        ++reads;
      }
      shared->reads.fetch_add(reads);
      std::_Exit(0);
    }
  }
  uint64_t generations[kRows]{};
  for (uint64_t iteration = 0; iteration < rounds; ++iteration) {
    const auto row = iteration % kRows;
    const Hash hash{row, 123};
    assert(directory.retire(hash) == static_cast<int64_t>(row));
    while (directory.pins(row)) usleep(1);
    const auto generation = ++generations[row];
    for (int word = 0; word < 8; ++word)
      shared->data[row][word].store(generation * 8 + word, std::memory_order_relaxed);
    directory.publish(row, hash);
    if (iteration % 4096 == 0) directory.rebuild();
  }
  shared->stop.store(true, std::memory_order_release);
  for (auto reader : readers) {
    int status;
    assert(waitpid(reader, &status, 0) == reader);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  assert(directory.pinned_refs() == 0 && directory.freeze());
  std::cout << "PASS: " << rounds * 2 << " publish/retire operations, "
            << shared->reads.load() << " validated concurrent reads\n";
  munmap(shared, sizeof(Shared));
  munmap(memory, Directory::bytes(kRows));
}
