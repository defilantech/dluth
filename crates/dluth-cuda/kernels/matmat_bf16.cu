// A BF16 matrix times a small batch of vectors:
//     y[b][r] = sum over c of w[r][c] * x[b][c]
// w is rows by cols, row after row, stored as BF16 (the top 16 bits of a 32-bit float).
// x is batch by cols and y is batch by rows, both 32-bit floats.
//
// The weights are read once and used for every vector of the batch.
//
// Each row is cut into `segs` equal runs of columns, one thread per run. The threads of a
// row sit next to each other in a warp (segs is a power of two up to 32 and the block size
// is a multiple of 32), and their partial sums are combined with warp shuffles.
//
// There is one kernel per batch size, matmat_bf16_b1 to matmat_bf16_b16, so that the loop
// over the batch has a fixed length and the compiler keeps the running sums in registers.
//
// `wide` says the caller checked what the fast path needs: cols is a multiple of 8 and all
// three pointers are 16-byte aligned. Then eight weights are read with one 16-byte load and
// four inputs with another, and the threads of a row take turns along it instead of each
// owning one run. Otherwise values are read one at a time. Both paths compute the same sums
// in a different order.
// CPU reference: dluth_kernels::cpu::matvec_bf16, once per vector of the batch.

template <int B>
__device__ __forceinline__ void matmat_bf16_body(const unsigned short* w, const float* x, float* y,
                                                 unsigned int rows, unsigned int cols,
                                                 unsigned int segs, unsigned int wide) {
    unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row = thread / segs;
    unsigned int seg = (unsigned int)(thread % segs);
    float acc[B];
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    if (row < rows) {
        unsigned int begin = (unsigned int)((unsigned long long)seg * cols / segs);
        unsigned int end = (unsigned int)((unsigned long long)(seg + 1) * cols / segs);
        const unsigned short* wr = w + row * cols;
        if (wide) {
            // Thread `seg` takes every segs-th group of eight columns, so at each step the
            // threads of a row read one unbroken stretch of memory between them.
            for (unsigned int c = seg * 8; c < cols; c += segs * 8) {
                uint4 bits = *reinterpret_cast<const uint4*>(wr + c);
                float v0 = __uint_as_float(bits.x << 16), v1 = __uint_as_float(bits.x & 0xffff0000u);
                float v2 = __uint_as_float(bits.y << 16), v3 = __uint_as_float(bits.y & 0xffff0000u);
                float v4 = __uint_as_float(bits.z << 16), v5 = __uint_as_float(bits.z & 0xffff0000u);
                float v6 = __uint_as_float(bits.w << 16), v7 = __uint_as_float(bits.w & 0xffff0000u);
                for (int b = 0; b < B; ++b) {
                    const float4* xb = reinterpret_cast<const float4*>(x + (size_t)b * cols + c);
                    float4 lo = xb[0], hi = xb[1];
                    acc[b] += v0 * lo.x + v1 * lo.y + v2 * lo.z + v3 * lo.w
                            + v4 * hi.x + v5 * hi.y + v6 * hi.z + v7 * hi.w;
                }
            }
        } else {
            for (unsigned int c = begin; c < end; ++c) {
                float v = __uint_as_float((unsigned int)wr[c] << 16);
                for (int b = 0; b < B; ++b) acc[b] += v * x[(size_t)b * cols + c];
            }
        }
    }
    // Every thread of the warp takes part in the shuffles, including those past the last row.
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2)
        for (int b = 0; b < B; ++b) acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], offset);
    if (seg == 0 && row < rows)
        for (int b = 0; b < B; ++b) y[(size_t)b * rows + row] = acc[b];
}

#define MATMAT_BF16(B)                                                                             \
    extern "C" __global__ void matmat_bf16_b##B(const unsigned short* w, const float* x, float* y, \
                                                unsigned int rows, unsigned int cols,              \
                                                unsigned int segs, unsigned int wide) {            \
        matmat_bf16_body<B>(w, x, y, rows, cols, segs, wide);                                      \
    }

MATMAT_BF16(1) MATMAT_BF16(2) MATMAT_BF16(3) MATMAT_BF16(4)
MATMAT_BF16(5) MATMAT_BF16(6) MATMAT_BF16(7) MATMAT_BF16(8)
MATMAT_BF16(9) MATMAT_BF16(10) MATMAT_BF16(11) MATMAT_BF16(12)
MATMAT_BF16(13) MATMAT_BF16(14) MATMAT_BF16(15) MATMAT_BF16(16)

// The same product for a matrix of few, long rows (a 64 by 4608 gate, a 128 by 2304
// router). With at most 32 threads to a row, as above, each thread walks hundreds of
// columns one load after another, and the kernel's time is that walk, not the bytes. Here
// a whole block of 256 threads shares one row for one vector: thread t takes every 256th
// group of eight columns, the sums are added within each warp by shuffles, and one thread
// adds the eight warps'.
// Needs what the fast path above needs: cols a multiple of 8, the pointers 16-byte aligned.
extern "C" __global__ void matmat_bf16_short(const unsigned short* w, const float* x, float* y,
                                             unsigned int rows, unsigned int cols, unsigned int batch) {
    __shared__ float part[8];
    unsigned int b = blockIdx.x / rows;
    unsigned int row = blockIdx.x % rows;
    if (b >= batch) return;
    const unsigned short* wr = w + (size_t)row * cols;
    const float* xv = x + (size_t)b * cols;
    float acc = 0.0f;
    for (unsigned int c = threadIdx.x * 8; c < cols; c += blockDim.x * 8) {
        uint4 bits = *reinterpret_cast<const uint4*>(wr + c);
        const float4* xb = reinterpret_cast<const float4*>(xv + c);
        float4 lo = xb[0], hi = xb[1];
        acc += __uint_as_float(bits.x << 16) * lo.x + __uint_as_float(bits.x & 0xffff0000u) * lo.y
             + __uint_as_float(bits.y << 16) * lo.z + __uint_as_float(bits.y & 0xffff0000u) * lo.w
             + __uint_as_float(bits.z << 16) * hi.x + __uint_as_float(bits.z & 0xffff0000u) * hi.y
             + __uint_as_float(bits.w << 16) * hi.z + __uint_as_float(bits.w & 0xffff0000u) * hi.w;
    }
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned int i = 0; i < blockDim.x / 32; ++i) total += part[i];
        y[(size_t)b * rows + row] = total;
    }
}
