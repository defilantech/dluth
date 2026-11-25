// Matrix times vector with the matrix in a block-scaled format, for a list of jobs.
//
// The feed-forward part of a layer is a set of expert networks, and each position uses a
// few of them. One job is one (expert, position) pair: "multiply this expert's matrix by
// that position's vector". All jobs of a layer run in one launch.
//
//     y[j][r] = factor[j] * sum over c of w_j[r][c] * x_j[c]
//
// Each job is three 64-bit words: where its matrix's values start (bytes into `values`),
// where its scales start (bytes into `scales`), and a word holding where its vector starts
// (floats into `x`, low 32 bits) and its factor (the bits of a float, high 32 bits).
//
// A row is cut into blocks that share one scale. Thread `seg` of a row takes every
// segs-th block, so the threads of a row read one unbroken stretch of memory between them
// (segs is a power of two up to 32, the block size of the launch a multiple of 32), and
// their partial sums are combined with warp shuffles.

// NVFP4: 4-bit codes, two per byte with the earlier weight in the low half; one 8-bit float
// scale (E4M3, looked up in `e4m3`, a table of 256 floats) per block of 16 weights.
// `pairs` is a table of 256 pairs of floats: for each possible byte, the values of its two
// codes, earlier weight first. Looking a byte up is far cheaper than decoding its codes
// with arithmetic: the first version of this kernel did that and read 13 times slower.
// cols is a multiple of 16; `values` and each job's offset are 8-byte aligned and `x` and
// each job's vector 16-byte aligned.
// CPU reference: dluth_kernels::blockfloat::matvec_nvfp4, once per job.
extern "C" __global__ void matvec_nvfp4_jobs(const unsigned char* values, const unsigned char* scales,
                                             const unsigned long long* jobs, const float* e4m3,
                                             const float2* pairs, const float* x, float* y,
                                             unsigned int rows, unsigned int cols,
                                             unsigned int n_jobs, unsigned int segs) {
    unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long group = thread / segs;
    unsigned int seg = (unsigned int)(thread % segs);
    unsigned long long job = group / rows;
    unsigned int row = (unsigned int)(group % rows);
    float acc = 0.0f;
    bool live = job < n_jobs;
    float factor = 0.0f;
    if (live) {
        unsigned long long word = jobs[3 * job + 2];
        factor = __uint_as_float((unsigned int)(word >> 32));
        const float* xj = x + (unsigned int)word;
        unsigned int blocks = cols / 16;
        const unsigned long long* codes =
            reinterpret_cast<const unsigned long long*>(values + jobs[3 * job] + (size_t)row * (cols / 2));
        const unsigned char* row_scales = scales + jobs[3 * job + 1] + (size_t)row * blocks;
        for (unsigned int k = seg; k < blocks; k += segs) {
            unsigned long long bits = codes[k];
            const float4* xb = reinterpret_cast<const float4*>(xj + 16 * k);
            float partial = 0.0f;
            for (int q = 0; q < 4; ++q) {
                float4 v = xb[q];
                float2 first = pairs[(unsigned int)(bits >> (16 * q)) & 255u];
                float2 second = pairs[(unsigned int)(bits >> (16 * q + 8)) & 255u];
                partial += first.x * v.x + first.y * v.y + second.x * v.z + second.y * v.w;
            }
            acc += partial * e4m3[row_scales[k]];
        }
    }
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (live && seg == 0) y[job * rows + row] = acc * factor;
}

// MXFP8: one 8-bit float (E4M3, looked up in `e4m3`) per weight; one power-of-two scale
// (E8M0, looked up in `e8m0`, a table of 256 floats) per block of 32 weights.
// cols is a multiple of 32; `values`, `x` and each job's offsets are 16-byte aligned.
// The job's factor is not used by this format's files; it is applied all the same.
// CPU reference: dluth_kernels::blockfloat::matvec_mxfp8, once per job.
extern "C" __global__ void matvec_mxfp8_jobs(const unsigned char* values, const unsigned char* scales,
                                             const unsigned long long* jobs, const float* e4m3,
                                             const float* e8m0, const float* x, float* y,
                                             unsigned int rows, unsigned int cols,
                                             unsigned int n_jobs, unsigned int segs) {
    unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long group = thread / segs;
    unsigned int seg = (unsigned int)(thread % segs);
    unsigned long long job = group / rows;
    unsigned int row = (unsigned int)(group % rows);
    float acc = 0.0f;
    bool live = job < n_jobs;
    float factor = 0.0f;
    if (live) {
        unsigned long long word = jobs[3 * job + 2];
        factor = __uint_as_float((unsigned int)(word >> 32));
        const float* xj = x + (unsigned int)word;
        unsigned int blocks = cols / 32;
        const uint4* bytes = reinterpret_cast<const uint4*>(values + jobs[3 * job] + (size_t)row * cols);
        const unsigned char* row_scales = scales + jobs[3 * job + 1] + (size_t)row * blocks;
        for (unsigned int k = seg; k < blocks; k += segs) {
            const float4* xb = reinterpret_cast<const float4*>(xj + 32 * k);
            float partial = 0.0f;
            for (int half = 0; half < 2; ++half) {
                uint4 b = bytes[2 * k + half];
                unsigned int words[4] = {b.x, b.y, b.z, b.w};
                for (int q = 0; q < 4; ++q) {
                    float4 v = xb[4 * half + q];
                    unsigned int four = words[q];
                    partial += e4m3[four & 255u] * v.x + e4m3[(four >> 8) & 255u] * v.y
                             + e4m3[(four >> 16) & 255u] * v.z + e4m3[four >> 24] * v.w;
                }
            }
            acc += partial * e8m0[row_scales[k]];
        }
    }
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (live && seg == 0) y[job * rows + row] = acc * factor;
}
