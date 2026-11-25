// Probes: kernels that measure the machine rather than compute anything for a model.
// They answer two questions the design depends on: how fast can a kernel read memory,
// and what does it cost to launch a kernel at all.

// Fills data[i] with a value that depends on i, so that a sum over any range is known
// in advance and a kernel that skips or repeats elements gives a different sum.
// CPU reference: dluth_cuda::probe::pattern.
extern "C" __global__ void probe_fill_u32(unsigned int* data, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) data[i] = (unsigned int)i * 2654435761u + 12345u;
}

// Sums n 32-bit values into 64-bit partial sums, one per block. Block b owns the
// elements [b * per_block, (b + 1) * per_block); its threads take every blockDim.x-th
// element, so neighbouring threads read neighbouring memory. `partial` must be zero
// before the launch. CPU reference: the wrapping 64-bit sum of the pattern.
extern "C" __global__ void probe_sum_u32(const unsigned int* data, unsigned long long n,
                                         unsigned long long per_block, unsigned long long* partial) {
    unsigned long long begin = (unsigned long long)blockIdx.x * per_block;
    unsigned long long end = begin + per_block;
    if (end > n) end = n;
    unsigned long long local = 0;
    for (unsigned long long i = begin + threadIdx.x; i < end; i += blockDim.x) local += data[i];
    atomicAdd(&partial[blockIdx.x], local);
}

// Does nothing. Launching it measures the fixed cost of a launch.
extern "C" __global__ void probe_empty() {}
