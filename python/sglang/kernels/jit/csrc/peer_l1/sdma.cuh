#pragma once

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "mori/application/transport/sdma/anvil.hpp"
#include <stdexcept>
#include <string>

namespace sglang::peer_l1 {

// Same layout as copy.cuh's Descriptor and Job: the copier passes one table to either engine.
struct SdmaDescriptor {
  uint64_t local, bytes, peer[8];
};
struct SdmaJob {
  int64_t peer, source, destination;
};

// Packets per ring reservation; 2048 * 28 B stays well inside anvil's 256 KB ring.
constexpr int64_t kSdmaChunk = 2048;

// One wave per queue; queue q pulls jobs [job_begin[q], job_begin[q + 1]), all from its peer.
// For each layer group it writes the copy-linear packets of the group's (descriptor,
// job) pairs, then one atomic increment of the queue's signal, so group g of a batch
// has landed on queue q once the signal reaches the batch's base + g + 1. Lane 0
// reserves ring space and rings the doorbell; all lanes write packets.
__global__ void sdma_submit_kernel(
    anvil::SdmaQueueDeviceHandle* const* handles,
    const int64_t* job_begin,
    const SdmaDescriptor* desc,
    const SdmaJob* jobs,
    const int64_t* group_first,
    int64_t groups,
    HSAuint64* const* signals) {
  const int64_t q = blockIdx.x;
  const int lane = threadIdx.x;
  __shared__ uint64_t s_base, s_pad;
  anvil::SdmaQueueDeviceHandle h = *handles[q];
  const int64_t j0 = job_begin[q];
  const int64_t njobs = job_begin[q + 1] - j0;
  constexpr uint32_t kCopyDw = sizeof(SDMA_PKT_COPY_LINEAR) / sizeof(uint32_t);
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t d0 = group_first[g];
    const int64_t total = njobs * (group_first[g + 1] - d0);
    int64_t i = 0;
    do {
      const int64_t n = total - i < kSdmaChunk ? total - i : kSdmaChunk;
      const bool last = i + n == total;
      const size_t bytes = n * sizeof(SDMA_PKT_COPY_LINEAR) + (last ? sizeof(SDMA_PKT_ATOMIC) : 0);
      if (lane == 0) {
        uint64_t pad = 0;
        s_base = h.ReserveQueueSpace(bytes, pad);
        s_pad = pad;
        // Wrap padding: zero dwords are NOPs.
        const uint64_t first_dw = h.WrapIntoRing(s_base) / sizeof(uint32_t);
        for (uint64_t d = 0; d < pad / sizeof(uint32_t); ++d)
          __hip_atomic_store(h.queueBuf + first_dw + d, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
      const uint64_t start = s_base + s_pad;
      for (int64_t k = lane; k < n; k += blockDim.x) {
        const int64_t idx = i + k;
        const SdmaDescriptor& d = desc[d0 + idx / njobs];
        const SdmaJob& job = jobs[j0 + idx % njobs];
        auto pkt = anvil::CreateCopyPacket(
            reinterpret_cast<void*>(d.peer[job.peer] + job.source * d.bytes),
            reinterpret_cast<void*>(d.local + job.destination * d.bytes),
            d.bytes);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&pkt);
        const uint64_t dw = h.WrapIntoRing(start + k * sizeof(SDMA_PKT_COPY_LINEAR)) / sizeof(uint32_t);
        for (uint32_t x = 0; x < kCopyDw; ++x)
          __hip_atomic_store(h.queueBuf + dw + x, w[x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
      if (lane == 0) {
        uint64_t wptr = start + n * sizeof(SDMA_PKT_COPY_LINEAR);
        if (last) {
          auto sig = anvil::CreateAtomicIncPacket(signals[q]);
          h.placePacket(sig, wptr, 0);
        }
        h.submitPacket(s_base, wptr);
      }
      __syncthreads();
      i += n;
    } while (i < total);
  }
}

inline void check_sdma(hipError_t result, const char* what) {
  if (result != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(result));
}

// Connects `channels` queues from this GPU to each peer and writes their device handle
// pointers into handles, peer-major. peers is CPU int64 [P], handles CPU int64 [P * channels].
inline void sdma_connect(
    int64_t local, tvm::ffi::TensorView peers, int64_t channels, tvm::ffi::TensorView handles) {
  static bool initialized = false;
  int can_wait = 0;
  check_sdma(
      hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, static_cast<int>(local)),
      "hipDeviceGetAttribute(CanUseStreamWaitValue)");
  if (!can_wait) throw std::runtime_error("peer_l1 sdma needs hipStreamWaitValue64 support");
  if (!initialized) {
    anvil::anvil.init();
    initialized = true;
  }
  const auto* p = static_cast<const int64_t*>(peers.data_ptr());
  auto* out = static_cast<int64_t*>(handles.data_ptr());
  if (handles.numel() != peers.numel() * channels) throw std::invalid_argument("handles must hold peers * channels");
  for (int64_t i = 0; i < peers.numel(); ++i) {
    const int peer = static_cast<int>(p[i]);
    if (!anvil::anvil.connect(static_cast<int>(local), peer, static_cast<int>(channels)))
      throw std::runtime_error("anvil connect failed for peer " + std::to_string(peer));
    for (int64_t c = 0; c < channels; ++c) {
      auto* queue = anvil::anvil.getSdmaQueue(static_cast<int>(local), peer, static_cast<int>(c));
      if (queue == nullptr) throw std::runtime_error("no SDMA queue for peer " + std::to_string(peer));
      out[i * channels + c] = reinterpret_cast<int64_t>(queue->deviceHandle());
    }
  }
}

// One zeroed signal word per entry of signals (CPU int64 [Q]); hipStreamWaitValue64 needs
// hipMallocSignalMemory, which only accepts 8-byte allocations.
inline void sdma_alloc_signals(tvm::ffi::TensorView signals) {
  auto* out = static_cast<int64_t*>(signals.data_ptr());
  for (int64_t q = 0; q < signals.numel(); ++q) {
    void* ptr = nullptr;
    check_sdma(hipExtMallocWithFlags(&ptr, sizeof(HSAuint64), hipMallocSignalMemory), "hipExtMallocWithFlags");
    check_sdma(hipMemset(ptr, 0, sizeof(HSAuint64)), "hipMemset(signal)");
    out[q] = reinterpret_cast<int64_t>(ptr);
  }
  check_sdma(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

inline void sdma_submit(
    tvm::ffi::TensorView handles,
    tvm::ffi::TensorView job_begin,
    tvm::ffi::TensorView desc,
    tvm::ffi::TensorView jobs,
    tvm::ffi::TensorView group_first,
    int64_t groups,
    tvm::ffi::TensorView signals,
    int64_t stream) {
  if (desc.ndim() != 2 || desc.size(1) != 10 || jobs.ndim() != 2 || jobs.size(1) != 3 ||
      job_begin.numel() != handles.numel() + 1 || group_first.numel() != groups + 1 ||
      signals.numel() != handles.numel())
    throw std::invalid_argument("invalid peer L1 SDMA submission");
  hipLaunchKernelGGL(
      sdma_submit_kernel,
      dim3(handles.numel()),
      dim3(64),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<anvil::SdmaQueueDeviceHandle* const*>(handles.data_ptr()),
      static_cast<const int64_t*>(job_begin.data_ptr()),
      static_cast<const SdmaDescriptor*>(desc.data_ptr()),
      static_cast<const SdmaJob*>(jobs.data_ptr()),
      static_cast<const int64_t*>(group_first.data_ptr()),
      groups,
      static_cast<HSAuint64* const*>(signals.data_ptr()));
  check_sdma(hipGetLastError(), "sdma_submit_kernel launch");
}

// The stream waits until the signal word reaches value (greater or equal).
inline void sdma_wait(int64_t stream, int64_t signal, int64_t value) {
  check_sdma(
      hipStreamWaitValue64(
          reinterpret_cast<hipStream_t>(stream),
          reinterpret_cast<void*>(signal),
          static_cast<uint64_t>(value),
          hipStreamWaitValueGte,
          ~0ULL),
      "hipStreamWaitValue64");
}

}  // namespace sglang::peer_l1
