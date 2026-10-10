#pragma once

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "directory.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace sglang::peer_l1 {

// One descriptor per (logical layer, pool). Fixed eight-peer width keeps the
// hot loop independent of the model's number of physical pools.
struct Descriptor {
  uint64_t local, bytes, peer[8];
};
struct Job {
  int64_t peer, source, destination;
};

template <bool Verify>
__global__ void copy_rows(
    const Descriptor* desc, const Job* jobs, int64_t count, int64_t first, int64_t last, unsigned long long* error) {
  for (int64_t work = blockIdx.x; work < (last - first) * count; work += gridDim.x) {
    const auto d = desc[first + work / count];
    const auto job = jobs[work % count];
    const auto* source = reinterpret_cast<const uint4*>(d.peer[job.peer] + job.source * d.bytes);
    auto* destination = reinterpret_cast<uint4*>(d.local + job.destination * d.bytes);
    for (uint64_t i = threadIdx.x; i < d.bytes / sizeof(uint4); i += blockDim.x) {
      const auto value = source[i];
      if constexpr (Verify) {
        const auto actual = destination[i];
        if (value.x != actual.x || value.y != actual.y || value.z != actual.z || value.w != actual.w)
          atomicCAS(error, 0ULL, static_cast<unsigned long long>((first * count + work) + 1));
      } else {
        destination[i] = value;
      }
    }
  }
}

inline void check_hip(hipError_t result) {
  if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}

inline void launch_copy(
    tvm::ffi::TensorView desc,
    tvm::ffi::TensorView jobs,
    int64_t first,
    int64_t last,
    int64_t blocks,
    tvm::ffi::TensorView error,
    bool verify,
    int64_t stream) {
  if (desc.ndim() != 2 || desc.size(1) != 10 || jobs.ndim() != 2 || jobs.size(1) != 3 || first < 0 ||
      last > desc.size(0) || first > last || blocks <= 0)
    throw std::invalid_argument("invalid peer L1 copy geometry");
  const auto count = jobs.size(0);
  if (!count || first == last) return;
  auto kernel = verify ? copy_rows<true> : copy_rows<false>;
  kernel<<<std::min(blocks, (last - first) * count), 256, 0, reinterpret_cast<hipStream_t>(stream)>>>(
      static_cast<const Descriptor*>(desc.data_ptr()),
      static_cast<const Job*>(jobs.data_ptr()),
      count,
      first,
      last,
      static_cast<unsigned long long*>(error.data_ptr()));
  check_hip(hipGetLastError());
}

struct UnpinBatch {
  struct Reservation {
    void* directory;
    std::vector<int64_t> rows;
  };
  // Owned by the callback: no Python references, GIL, tensor destructors, or
  // HIP calls on that thread. Update each directory's global count in bulk.
  std::vector<Reservation> reservations;
};

inline void enqueue_unpin(tvm::ffi::TensorView entries, int64_t stream) {
  if (entries.ndim() != 2 || entries.size(1) != 2 || entries.device().device_type != kDLCPU)
    throw std::invalid_argument("peer L1 unpin entries must be a CPU [N,2] tensor");
  auto batch = std::make_unique<UnpinBatch>();
  const auto* data = static_cast<const int64_t*>(entries.data_ptr());
  for (int64_t i = 0; i < entries.numel(); i += 2) {
    auto* directory = reinterpret_cast<void*>(data[i]);
    if (batch->reservations.empty() || batch->reservations.back().directory != directory)
      batch->reservations.push_back({directory, {}});
    batch->reservations.back().rows.push_back(data[i + 1]);
  }
  check_hip(hipLaunchHostFunc(
      reinterpret_cast<hipStream_t>(stream),
      [](void* context) {
        std::unique_ptr<UnpinBatch> batch(static_cast<UnpinBatch*>(context));
        for (const auto& reservation : batch->reservations)
          Directory(reservation.directory).unpin(reservation.rows.data(), reservation.rows.size());
      },
      batch.get()));
  batch.release();
}

}  // namespace sglang::peer_l1
