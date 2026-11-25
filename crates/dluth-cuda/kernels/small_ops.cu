// Small operations on whole vectors. None reads many bytes; they exist so that a token's
// values can stay on the device from one matrix product to the next.

// out[i] = a[i] + b[i].
extern "C" __global__ void vec_add(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// Root-mean-square normalization with learned weights, for a batch of vectors of n
// floats, one block of threads per vector:
//     out[b][i] = x[b][i] / sqrt(mean(x[b]^2) + eps) * w[i]
// CPU reference: cpu::rmsnorm, once per vector. The squares are added in a different order.
extern "C" __global__ void rmsnorm_weighted(const float* x, const float* w, float* out, unsigned int n,
                                            float eps, unsigned int batch) {
    __shared__ float part[1024];
    if (blockIdx.x >= batch) return;
    const float* v = x + (size_t)blockIdx.x * n;
    float* o = out + (size_t)blockIdx.x * n;
    float squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) squares += v[i] * v[i];
    part[threadIdx.x] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) squares += part[t];
        part[0] = squares;
    }
    __syncthreads();
    float scale = 1.0f / sqrtf(part[0] / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) o[i] = v[i] * scale * w[i];
}

// Scores for a blend across depth: for each of `count` candidates of `width` floats, one
// after another in `candidates`, scores[j] = (query . candidate_j) * scale. One block of
// threads per candidate. CPU reference: the scoring in depthmix::mix_normalized.
extern "C" __global__ void blend_scores(const float* query, const float* candidates, float* scores,
                                        unsigned int count, unsigned int width, float scale) {
    __shared__ float part[1024];
    unsigned int j = blockIdx.x;
    if (j >= count) return;
    const float* c = candidates + (size_t)j * width;
    float dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) dot += query[i] * c[i];
    part[threadIdx.x] = dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) dot += part[t];
        scores[j] = dot * scale;
    }
}

// The gate that merges a message from the previous pass into a sublayer input:
//     out[i] = (1 - tanh(forget[i])) * x[i] + tanh(gain[i]) * message[i]
// CPU reference: the last step of the model's merge.
extern "C" __global__ void gate_merge(const float* x, const float* message, const float* forget,
                                      const float* gain, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.0f - tanhf(forget[i])) * x[i] + tanhf(gain[i]) * message[i];
}

// x[i] = cap * tanh(x[i] / cap), in place. CPU reference: cpu::softcap.
extern "C" __global__ void softcap_inplace(float* x, unsigned int n, float cap) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// Most candidates a blend takes in one launch (see blend_many).
#define BLEND_MAX 16

// Up to eight candidates, each wherever it lives and each a batch of vectors of `width`,
// normalized into one buffer in one launch:
//     out[b][j][i] = x_j[b][i] / sqrt(mean(x_j[b]^2) + eps)
// so that position b's candidates end up one after another. One block of threads per
// vector. CPU reference: cpu::rmsnorm_plain, once per vector.
extern "C" __global__ void normalize_many(const float* x0, const float* x1, const float* x2, const float* x3,
                                          const float* x4, const float* x5, const float* x6, const float* x7,
                                          float* out, unsigned int count, unsigned int width, float eps,
                                          unsigned int batch) {
    __shared__ float part[1024];
    unsigned int j = blockIdx.x % count;
    unsigned int b = blockIdx.x / count;
    if (b >= batch) return;
    const float* all[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
    const float* x = all[j] + (size_t)b * width;
    float squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) squares += x[i] * x[i];
    part[threadIdx.x] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) squares += part[t];
        part[0] = squares;
    }
    __syncthreads();
    float scale = 1.0f / sqrtf(part[0] / (float)width + eps);
    float* o = out + ((size_t)b * count + j) * width;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) o[i] = x[i] * scale;
}

// out[i] = x[i]: one position of a batch copied out as a batch of one.
extern "C" __global__ void vec_copy(const float* x, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

// A whole blend across depth for up to four queries at once, from the candidates as they
// are (not normalized beforehand), each optionally followed by a normalization with
// learned weights. For position b of a batch and query q:
//     n_j     = candidate_j[b] / sqrt(mean(candidate_j[b]^2) + blend_eps)
//     weights = softmax over j of (query_q . n_j) * scale
//     mixed   = sum over j of weights_j * n_j
//     out_q[b] = mixed                                           if norm_q is null
//              = mixed / sqrt(mean(mixed^2) + norm_eps) * norm_q  otherwise
// One block of threads per (position, query). Each candidate is its own buffer holding the
// batch; each query and each norm is one vector of `width`; each output holds the batch.
// CPU reference: depthmix::normalize, depthmix::mix_normalized and cpu::rmsnorm, one after
// the other. Here a candidate's dot product with the query is taken before the candidate
// is scaled, and the sums are added in a different order.
// The sum of `v` over the 32 threads of a warp, in every one of them. Every thread of the
// warp must call it.
__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// Sums are first taken within each warp of 32 threads, by shuffles, and one thread then
// adds the few warps' sums: a block's single thread adding every thread's part one after
// the other took most of this kernel's time.
#define BLEND_QUERIES 4
#define BLEND_WARPS 8
extern "C" __global__ void blend_many(
    const float* c0, const float* c1, const float* c2, const float* c3, const float* c4, const float* c5,
    const float* c6, const float* c7, const float* c8, const float* c9, const float* c10, const float* c11,
    const float* c12, const float* c13, const float* c14, const float* c15,
    const float* q0, const float* q1, const float* q2, const float* q3,
    const float* w0, const float* w1, const float* w2, const float* w3,
    float* o0, float* o1, float* o2, float* o3,
    unsigned int count, unsigned int width, float scale, float blend_eps, float norm_eps,
    unsigned int batch, unsigned int queries) {
    __shared__ float squares[BLEND_MAX][BLEND_WARPS];
    __shared__ float dots[BLEND_MAX][BLEND_WARPS];
    __shared__ float coefficient[BLEND_MAX];
    unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    unsigned int b = blockIdx.x / queries;
    unsigned int which = blockIdx.x % queries;
    if (b >= batch) return;
    const float* candidates[BLEND_MAX] = {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15};
    const float* all_queries[BLEND_QUERIES] = {q0, q1, q2, q3};
    const float* all_norms[BLEND_QUERIES] = {w0, w1, w2, w3};
    float* all_outs[BLEND_QUERIES] = {o0, o1, o2, o3};
    const float* query = all_queries[which];
    const float* norm = all_norms[which];
    float* out = all_outs[which] + (size_t)b * width;
    size_t at = (size_t)b * width;

    float mine_squares[BLEND_MAX];
    float mine_dots[BLEND_MAX];
    for (unsigned int j = 0; j < count; ++j) mine_squares[j] = mine_dots[j] = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float q = query[i];
        for (unsigned int j = 0; j < count; ++j) {
            float x = candidates[j][at + i];
            mine_squares[j] += x * x;
            mine_dots[j] += q * x;
        }
    }
    for (unsigned int j = 0; j < count; ++j) {
        float s = warp_sum(mine_squares[j]);
        float d = warp_sum(mine_dots[j]);
        if (lane == 0) {
            squares[j][warp] = s;
            dots[j][warp] = d;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = -INFINITY;
        for (unsigned int j = 0; j < count; ++j) {
            float s = 0.0f, d = 0.0f;
            for (unsigned int t = 0; t < warps; ++t) {
                s += squares[j][t];
                d += dots[j][t];
            }
            float inverse = 1.0f / sqrtf(s / (float)width + blend_eps);
            squares[j][0] = inverse;
            dots[j][0] = d * inverse * scale;
            best = fmaxf(best, dots[j][0]);
        }
        float sum = 0.0f;
        for (unsigned int j = 0; j < count; ++j) {
            coefficient[j] = expf(dots[j][0] - best);
            sum += coefficient[j];
        }
        // The weight of a candidate times the factor that normalizes it.
        for (unsigned int j = 0; j < count; ++j) coefficient[j] = coefficient[j] / sum * squares[j][0];
    }
    __syncthreads();
    float mixed_squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float total = 0.0f;
        for (unsigned int j = 0; j < count; ++j) total += coefficient[j] * candidates[j][at + i];
        out[i] = total;
        mixed_squares += total * total;
    }
    if (norm == nullptr) return;
    mixed_squares = warp_sum(mixed_squares);
    __syncthreads();
    if (lane == 0) squares[0][warp] = mixed_squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned int t = 0; t < warps; ++t) s += squares[0][t];
        coefficient[0] = 1.0f / sqrtf(s / (float)width + norm_eps);
    }
    __syncthreads();
    float factor = coefficient[0];
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) out[i] = out[i] * factor * norm[i];
}
