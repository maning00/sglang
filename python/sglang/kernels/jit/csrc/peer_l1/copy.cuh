#pragma once

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

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

struct CleanupBatch {
  hipHostFn_t enqueue;
  std::vector<int64_t> items;
};

/// One callback enqueues all grants after every source read has completed.
inline void enqueue_cleanup(tvm::ffi::TensorView entries, int64_t callback, int64_t stream) {
  if (entries.ndim() != 1 || entries.device().device_type != kDLCPU)
    throw std::invalid_argument("peer L1 cleanup entries must be a CPU [N] tensor");
  const auto* data = static_cast<const int64_t*>(entries.data_ptr());
  auto batch = std::make_unique<CleanupBatch>();
  batch->enqueue = reinterpret_cast<hipHostFn_t>(callback);
  batch->items.assign(data, data + entries.numel());
  check_hip(hipLaunchHostFunc(
      reinterpret_cast<hipStream_t>(stream),
      [](void* context) {
        std::unique_ptr<CleanupBatch> batch(static_cast<CleanupBatch*>(context));
        for (auto item : batch->items)
          batch->enqueue(reinterpret_cast<void*>(item));
      },
      batch.get()));
  batch.release();
}

}  // namespace sglang::peer_l1
