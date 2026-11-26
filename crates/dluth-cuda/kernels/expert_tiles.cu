// Block-format matrices times many vectors on the tensor cores, for a list of jobs.
//
//     y[out][r] = factor[j] * sum over c of w_j[r][c] * t(x_v[c])     for each (v, out) of job j
//
// The same jobs as matvec_block.cu runs, for the case that pays here: an expert chosen by
// many positions at once, as when a prompt is read in wide sweeps. There a thread
// multiplies and adds one pair of numbers at a time. A tensor core multiplies a 16 by 16
// tile by another in one instruction, which is many times the arithmetic per second, and
// the matrix is still read from memory once per job.
//
// The tensor cores take 16-bit floats (IEEE half precision) and add up in 32-bit floats.
//   * The weights lose nothing. A 4-bit code times its 8-bit scale has at most 6
//     significant bits and a size well inside a half's range. An 8-bit weight times its
//     power of two is exact too when the power is from 2^-15 to 2^7, which the caller checks
//     (tiles.rs, exact_in_16_bits).
//   * An input goes in as two halfs: the half nearest to it, and the half nearest to what
//     that left over. Both are multiplied and added, so the input counts as their sum
//     t(x), which is x to about 22 significant bits where a 32-bit float has 24. (One half
//     alone, 11 bits, moved the real model's scores by more than its tests allow.) An
//     input beyond the largest half counts as the largest half, and one that is not a
//     number stays so. That, and the order of the additions, is all that differs from
//     matvec_block.cu.
//
// Two steps. tile_pack writes each job's inputs as halfs, n nearest halfs and then n
// leftovers for each column, zero for a vector past the job's last:
//     packed[job][c][v] = near(x_v[c]),   packed[job][c][n + v] = near(x_v[c] - near(x_v[c]))
// Then one warp per (job, strip of 16 rows) walks the strip 64 columns at a time: every
// lane reads 16 bytes of one row, the warp expands them to a 16 by 64 tile of halfs in
// shared memory, and four tile products add it into the sums. The instructions are
// NVIDIA's documented warp matrix functions (mma.h, CUDA C++ Programming Guide).
//
// A job is TILE_WORDS 64-bit words: where its matrix's values start (bytes into `values`);
// where its scales start (bytes into `scales`); a word holding how many vectors it has (low
// 32 bits) and its factor (the bits of a float, high 32 bits); where its packed inputs
// start (halfs into `packed`); then, for each of up to TILE_VECTORS vectors, a word holding
// where the vector starts (floats into `x`, low 32 bits) and which output it fills (high
// 32 bits; output `out` is y[out * rows ...]).
//
// A launch runs jobs of one width: 16, 32 or 64 vectors (NT tiles of 16), so that the sums
// stay in registers. A job with fewer vectors than its launch's width has zeros in the rest.
//
// rows is a multiple of 16 and cols of 64. `values` and each job's offset into it are
// 16-byte aligned; `packed` and each job's offset into it are 32-byte aligned, as the
// warp matrix functions ask of the memory they load from.
// CPU reference: dluth_kernels::blockfloat::matvec_nvfp4 and matvec_mxfp8, once per vector
// of each job, on the inputs as their two halfs give them.
#include <mma.h>
#include <cuda_fp16.h>

#define TILE_VECTORS 64
#define TILE_WORDS (4 + TILE_VECTORS)
// Warps to a block; each has its own tile in shared memory.
#define TILE_WARPS 4
// How many passes of 64 columns a lane reads before the warp works on them. One at a time,
// a warp waits for memory after every pass and the kernel read 103 GB/s; six ahead it read
// 132 GB/s; nine or more was slower again (more registers, fewer warps at once).
#define TILE_AHEAD 6

extern "C" __global__ void tile_pack(const unsigned long long* jobs, const float* x, __half* packed,
                                     unsigned int cols, unsigned int n_jobs, unsigned int n) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long job = index / cols;
    unsigned int c = (unsigned int)(index % cols);
    if (job >= n_jobs) return;
    const unsigned long long* rec = jobs + job * TILE_WORDS;
    unsigned int count = (unsigned int)rec[2];
    __half* out = packed + rec[3] + (size_t)c * (2 * n);
    for (unsigned int v = 0; v < n; ++v) {
        float f = 0.0f;
        if (v < count) {
            float given = x[(size_t)(unsigned int)rec[4 + v] + c];
            f = fminf(fmaxf(given, -65504.0f), 65504.0f);
            // The limits would turn something that is not a number into one.
            if (given != given) f = given;
        }
        __half near = __float2half(f);
        out[v] = near;
        // Exact: the leftover of a rounding is itself a float. It is at most half a step
        // of `near`, so it is a half too unless it is tinier than any half.
        out[n + v] = __float2half(f - __half2float(near));
    }
}

// The values of the sixteen 4-bit codes: bit 3 is the sign, the rest picks the size.
__device__ const float TILE_E2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

template <int NT>
struct TileSums {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[NT];
};

// Adds a tile of 16 rows by 64 columns times the job's inputs for those columns. `xin`
// points at the first of those columns in the job's packed inputs: for each column the n
// nearest halfs and then the n leftovers, which are added into the same sums.
template <int NT>
__device__ __forceinline__ void tile_add(TileSums<NT>& sums, const __half* tile, const __half* xin) {
    using namespace nvcuda;
    const unsigned int n = NT * 16;
    for (int k = 0; k < 4; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::load_matrix_sync(a, tile + k * 16, 64);
        for (int t = 0; t < 2 * NT; ++t) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
            wmma::load_matrix_sync(b, xin + (size_t)(k * 16) * (2 * n) + t * 16, 2 * n);
            wmma::mma_sync(sums.acc[t % NT], a, b, sums.acc[t % NT]);
        }
    }
}

// Writes the strip's sums, times the job's factor, to the outputs the job names. `spare`
// is the warp's tile, used as room for 16 by 16 floats.
template <int NT>
__device__ __forceinline__ void tile_store(TileSums<NT>& sums, __half* spare, const unsigned long long* rec,
                                           float* y, unsigned int rows, unsigned int row, unsigned int lane) {
    using namespace nvcuda;
    float* f = reinterpret_cast<float*>(spare);
    unsigned int count = (unsigned int)rec[2];
    float factor = __uint_as_float((unsigned int)(rec[2] >> 32));
    unsigned int r = lane / 2, part = lane % 2;
    for (int t = 0; t < NT; ++t) {
        wmma::store_matrix_sync(f, sums.acc[t], 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int i = 0; i < 8; ++i) {
            unsigned int in_tile = part * 8 + i, v = t * 16 + in_tile;
            if (v < count) y[(size_t)(unsigned int)(rec[4 + v] >> 32) * rows + row] = f[r * 16 + in_tile] * factor;
        }
        __syncwarp();
    }
}

// NVFP4: 4-bit codes, two per byte with the earlier weight in the low half; one 8-bit float
// scale (E4M3, looked up in `e4m3`, a table of 256 floats) per block of 16 weights. A pass
// of 64 columns is 32 bytes of a row's codes and 4 of its scales. Lane 2r takes the first
// half of row r of the strip (two blocks) and lane 2r + 1 the second.
template <int NT>
__device__ __forceinline__ void tiles_nvfp4_body(__half* tile, const unsigned char* values,
                                                 const unsigned char* scales, const unsigned long long* jobs,
                                                 const float* e4m3, const __half* packed, float* y,
                                                 unsigned int rows, unsigned int cols, unsigned int n_jobs) {
    using namespace nvcuda;
    unsigned int lane = threadIdx.x % 32;
    unsigned long long warp = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int strips = rows / 16;
    unsigned long long job = warp / strips;
    // The same for every lane of a warp, so a warp leaves or stays as one.
    if (job >= n_jobs) return;
    const unsigned long long* rec = jobs + job * TILE_WORDS;
    unsigned int r = lane / 2, part = lane % 2;
    unsigned int row = (unsigned int)(warp % strips) * 16 + r;
    unsigned int passes = cols / 64;
    const uint4* codes = reinterpret_cast<const uint4*>(values + rec[0] + (size_t)row * (cols / 2));
    const unsigned int* row_scales = reinterpret_cast<const unsigned int*>(scales + rec[1] + (size_t)row * (cols / 16));
    const __half* xin = packed + rec[3];
    __half2* out = reinterpret_cast<__half2*>(tile + r * 64 + part * 32);
    TileSums<NT> sums;
    for (int t = 0; t < NT; ++t) wmma::fill_fragment(sums.acc[t], 0.0f);
    for (unsigned int first = 0; first < passes; first += TILE_AHEAD) {
        uint4 read[TILE_AHEAD];
        unsigned int read_scales[TILE_AHEAD];
        for (int i = 0; i < TILE_AHEAD; ++i) {
            if (first + i < passes) {
                read[i] = codes[2 * (first + i) + part];
                read_scales[i] = row_scales[first + i];
            }
        }
        for (int i = 0; i < TILE_AHEAD && first + i < passes; ++i) {
            unsigned int two = read_scales[i] >> (16 * part);
            float scale[2] = {e4m3[two & 255u], e4m3[(two >> 8) & 255u]};
            unsigned int words[4] = {read[i].x, read[i].y, read[i].z, read[i].w};
            for (int q = 0; q < 4; ++q) {
                unsigned int bits = words[q];
                float s = scale[q / 2];
                for (int b = 0; b < 4; ++b)
                    out[4 * q + b] = __floats2half2_rn(TILE_E2M1[(bits >> (8 * b)) & 15u] * s,
                                                       TILE_E2M1[(bits >> (8 * b + 4)) & 15u] * s);
            }
            __syncwarp();
            tile_add<NT>(sums, tile, xin + (size_t)(first + i) * 64 * (NT * 32));
            __syncwarp();
        }
    }
    tile_store<NT>(sums, tile, rec, y, rows, row, lane);
}

// MXFP8: one 8-bit float (E4M3, looked up in `e4m3`) per weight; one power-of-two scale
// (E8M0, looked up in `e8m0`, a table of 256 floats) per block of 32 weights. A pass of 64
// columns is 64 bytes of a row and 2 of its scales. Lane 2r takes the first block of row r
// of the strip and lane 2r + 1 the second. Half as many passes are read ahead as above,
// because a pass is twice the bytes.
template <int NT>
__device__ __forceinline__ void tiles_mxfp8_body(__half* tile, const unsigned char* values,
                                                 const unsigned char* scales, const unsigned long long* jobs,
                                                 const float* e4m3, const float* e8m0, const __half* packed,
                                                 float* y, unsigned int rows, unsigned int cols,
                                                 unsigned int n_jobs) {
    using namespace nvcuda;
    const int ahead = TILE_AHEAD / 2;
    unsigned int lane = threadIdx.x % 32;
    unsigned long long warp = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int strips = rows / 16;
    unsigned long long job = warp / strips;
    if (job >= n_jobs) return;
    const unsigned long long* rec = jobs + job * TILE_WORDS;
    unsigned int r = lane / 2, part = lane % 2;
    unsigned int row = (unsigned int)(warp % strips) * 16 + r;
    unsigned int passes = cols / 64;
    const uint4* bytes = reinterpret_cast<const uint4*>(values + rec[0] + (size_t)row * cols);
    const unsigned short* row_scales =
        reinterpret_cast<const unsigned short*>(scales + rec[1] + (size_t)row * (cols / 32));
    const __half* xin = packed + rec[3];
    __half2* out = reinterpret_cast<__half2*>(tile + r * 64 + part * 32);
    TileSums<NT> sums;
    for (int t = 0; t < NT; ++t) wmma::fill_fragment(sums.acc[t], 0.0f);
    for (unsigned int first = 0; first < passes; first += ahead) {
        uint4 read[2 * ahead];
        unsigned int read_scales[ahead];
        for (int i = 0; i < ahead; ++i) {
            if (first + i < passes) {
                read[2 * i] = bytes[4 * (first + i) + 2 * part];
                read[2 * i + 1] = bytes[4 * (first + i) + 2 * part + 1];
                read_scales[i] = row_scales[first + i];
            }
        }
        for (int i = 0; i < ahead && first + i < passes; ++i) {
            float s = e8m0[(read_scales[i] >> (8 * part)) & 255u];
            unsigned int words[8] = {read[2 * i].x,     read[2 * i].y,     read[2 * i].z,     read[2 * i].w,
                                     read[2 * i + 1].x, read[2 * i + 1].y, read[2 * i + 1].z, read[2 * i + 1].w};
            for (int q = 0; q < 8; ++q) {
                unsigned int four = words[q];
                out[2 * q] = __floats2half2_rn(e4m3[four & 255u] * s, e4m3[(four >> 8) & 255u] * s);
                out[2 * q + 1] = __floats2half2_rn(e4m3[(four >> 16) & 255u] * s, e4m3[four >> 24] * s);
            }
            __syncwarp();
            tile_add<NT>(sums, tile, xin + (size_t)(first + i) * 64 * (NT * 32));
            __syncwarp();
        }
    }
    tile_store<NT>(sums, tile, rec, y, rows, row, lane);
}

#define TILES(N, NT)                                                                                \
    extern "C" __global__ void tiles_nvfp4_##N(const unsigned char* values, const unsigned char* scales, \
                                               const unsigned long long* jobs, const float* e4m3,   \
                                               const __half* packed, float* y, unsigned int rows,   \
                                               unsigned int cols, unsigned int n_jobs) {            \
        __shared__ __align__(32) __half tiles[TILE_WARPS][16 * 64];                                 \
        tiles_nvfp4_body<NT>(tiles[threadIdx.x / 32], values, scales, jobs, e4m3, packed, y, rows,  \
                             cols, n_jobs);                                                         \
    }                                                                                               \
    extern "C" __global__ void tiles_mxfp8_##N(const unsigned char* values, const unsigned char* scales, \
                                               const unsigned long long* jobs, const float* e4m3,   \
                                               const float* e8m0, const __half* packed, float* y,   \
                                               unsigned int rows, unsigned int cols,                \
                                               unsigned int n_jobs) {                               \
        __shared__ __align__(32) __half tiles[TILE_WARPS][16 * 64];                                 \
        tiles_mxfp8_body<NT>(tiles[threadIdx.x / 32], values, scales, jobs, e4m3, e8m0, packed, y,  \
                             rows, cols, n_jobs);                                                   \
    }

TILES(16, 1) TILES(32, 2) TILES(64, 4)
