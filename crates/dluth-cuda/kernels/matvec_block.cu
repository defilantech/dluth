// Matrix times vectors with the matrix in a block-scaled format, for a list of jobs.
//
// The feed-forward part of a layer is a set of expert networks, and each position uses a
// few of them. Neighbouring positions often use the same expert, so one job is one expert
// with every position that chose it: "multiply this matrix by these vectors". The matrix
// is read once per job, whatever the number of vectors. All jobs of a layer run in one launch.
//
//     y[out][r] = factor[j] * sum over c of w_j[r][c] * x_v[c]     for each (v, out) of job j
//
// Each job is JOB_WORDS 64-bit words: where its matrix's values start (bytes into
// `values`); where its scales start (bytes into `scales`); a word holding how many vectors
// it has (low 32 bits) and its factor (the bits of a float, high 32 bits); then, for each
// of up to JOB_VECTORS vectors, a word holding where the vector starts (floats into `x`,
// low 32 bits) and which output it fills (high 32 bits; output `out` is y[out * rows ...]).
//
// A row is cut into blocks that share one scale. Thread `seg` of a row takes every
// segs-th block, so the threads of a row read one unbroken stretch of memory between them
// (segs is a power of two up to 32, the block size of the launch a multiple of 32), and
// their partial sums are combined with warp shuffles.
#define JOB_VECTORS 8
#define JOB_WORDS (3 + JOB_VECTORS)

// NVFP4: 4-bit codes, two per byte with the earlier weight in the low half; one 8-bit float
// scale (E4M3, looked up in `e4m3`, a table of 256 floats) per block of 16 weights.
// `pairs` is a table of 256 pairs of floats: for each possible byte, the values of its two
// codes, earlier weight first. Looking a byte up is far cheaper than decoding its codes
// with arithmetic: the first version of this kernel did that and read 13 times slower.
// cols is a multiple of 16; `values` and each job's offset are 8-byte aligned and `x` and
// each job's vector 16-byte aligned.
// CPU reference: dluth_kernels::blockfloat::matvec_nvfp4, once per vector of each job.
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
    float acc[JOB_VECTORS];
    for (int v = 0; v < JOB_VECTORS; ++v) acc[v] = 0.0f;
    bool live = job < n_jobs;
    const unsigned long long* rec = jobs + (live ? job : 0) * JOB_WORDS;
    unsigned int count = live ? (unsigned int)rec[2] : 0u;
    if (live) {
        unsigned int blocks = cols / 16;
        const unsigned long long* codes =
            reinterpret_cast<const unsigned long long*>(values + rec[0] + (size_t)row * (cols / 2));
        const unsigned char* row_scales = scales + rec[1] + (size_t)row * blocks;
        for (unsigned int k = seg; k < blocks; k += segs) {
            // Decode the block once; it is then used for every vector of the job.
            unsigned long long bits = codes[k];
            float2 w[8];
            for (int q = 0; q < 8; ++q) w[q] = pairs[(unsigned int)(bits >> (8 * q)) & 255u];
            float scale = e4m3[row_scales[k]];
            for (unsigned int v = 0; v < count; ++v) {
                const float4* xb = reinterpret_cast<const float4*>(x + (unsigned int)rec[3 + v] + 16 * k);
                float partial = 0.0f;
                for (int q = 0; q < 4; ++q) {
                    float4 f = xb[q];
                    partial += w[2 * q].x * f.x + w[2 * q].y * f.y + w[2 * q + 1].x * f.z + w[2 * q + 1].y * f.w;
                }
                acc[v] += partial * scale;
            }
        }
    }
    // Every thread of the warp takes part in the shuffles, for every slot, whatever its job.
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2)
        for (int v = 0; v < JOB_VECTORS; ++v) acc[v] += __shfl_xor_sync(0xffffffffu, acc[v], offset);
    if (live && seg == 0) {
        float factor = __uint_as_float((unsigned int)(rec[2] >> 32));
        for (unsigned int v = 0; v < count; ++v)
            y[(size_t)(unsigned int)(rec[3 + v] >> 32) * rows + row] = acc[v] * factor;
    }
}

// MXFP8: one 8-bit float (E4M3, looked up in `e4m3`) per weight; one power-of-two scale
// (E8M0, looked up in `e8m0`, a table of 256 floats) per block of 32 weights.
// cols is a multiple of 32; `values`, `x` and each job's offsets are 16-byte aligned.
// The job's factor is not used by this format's files; it is applied all the same.
// CPU reference: dluth_kernels::blockfloat::matvec_mxfp8, once per vector of each job.
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
    float acc[JOB_VECTORS];
    for (int v = 0; v < JOB_VECTORS; ++v) acc[v] = 0.0f;
    bool live = job < n_jobs;
    const unsigned long long* rec = jobs + (live ? job : 0) * JOB_WORDS;
    unsigned int count = live ? (unsigned int)rec[2] : 0u;
    if (live) {
        unsigned int blocks = cols / 32;
        const uint4* bytes = reinterpret_cast<const uint4*>(values + rec[0] + (size_t)row * cols);
        const unsigned char* row_scales = scales + rec[1] + (size_t)row * blocks;
        for (unsigned int k = seg; k < blocks; k += segs) {
            // Decode the block once; it is then used for every vector of the job.
            float w[32];
            for (int half = 0; half < 2; ++half) {
                uint4 b = bytes[2 * k + half];
                unsigned int words[4] = {b.x, b.y, b.z, b.w};
                for (int q = 0; q < 4; ++q) {
                    unsigned int four = words[q];
                    w[16 * half + 4 * q] = e4m3[four & 255u];
                    w[16 * half + 4 * q + 1] = e4m3[(four >> 8) & 255u];
                    w[16 * half + 4 * q + 2] = e4m3[(four >> 16) & 255u];
                    w[16 * half + 4 * q + 3] = e4m3[four >> 24];
                }
            }
            float scale = e8m0[row_scales[k]];
            for (unsigned int v = 0; v < count; ++v) {
                const float4* xb = reinterpret_cast<const float4*>(x + (unsigned int)rec[3 + v] + 32 * k);
                float partial = 0.0f;
                for (int q = 0; q < 8; ++q) {
                    float4 f = xb[q];
                    partial += w[4 * q] * f.x + w[4 * q + 1] * f.y + w[4 * q + 2] * f.z + w[4 * q + 3] * f.w;
                }
                acc[v] += partial * scale;
            }
        }
    }
    // Every thread of the warp takes part in the shuffles, for every slot, whatever its job.
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2)
        for (int v = 0; v < JOB_VECTORS; ++v) acc[v] += __shfl_xor_sync(0xffffffffu, acc[v], offset);
    if (live && seg == 0) {
        float factor = __uint_as_float((unsigned int)(rec[2] >> 32));
        for (unsigned int v = 0; v < count; ++v)
            y[(size_t)(unsigned int)(rec[3 + v] >> 32) * rows + row] = acc[v] * factor;
    }
}
