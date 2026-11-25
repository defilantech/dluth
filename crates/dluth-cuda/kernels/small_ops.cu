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

// A whole blend across depth in one launch, one block of threads per position of a batch:
//     score_j = (query . candidate_j) * scale;  weights = softmax(scores);
//     out[b] = sum over j of weights_j * candidate_j
// Position b's `count` candidates of `width` floats are one after another in `candidates`,
// after those of the positions before it. CPU reference: depthmix::mix_normalized, once per
// position.
#define BLEND_MAX 16
extern "C" __global__ void blend_mix(const float* query, const float* candidates, float* out,
                                     unsigned int count, unsigned int width, float scale,
                                     unsigned int batch) {
    __shared__ float part[BLEND_MAX][256];
    __shared__ float weight[BLEND_MAX];
    if (blockIdx.x >= batch) return;
    const float* mine = candidates + (size_t)blockIdx.x * count * width;
    float* o = out + (size_t)blockIdx.x * width;
    float dot[BLEND_MAX];
    for (unsigned int j = 0; j < count; ++j) dot[j] = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float q = query[i];
        for (unsigned int j = 0; j < count; ++j) dot[j] += q * mine[(size_t)j * width + i];
    }
    for (unsigned int j = 0; j < count; ++j) part[j][threadIdx.x] = dot[j];
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = -INFINITY;
        for (unsigned int j = 0; j < count; ++j) {
            float s = 0.0f;
            for (unsigned int t = 0; t < blockDim.x; ++t) s += part[j][t];
            weight[j] = s * scale;
            best = fmaxf(best, weight[j]);
        }
        float sum = 0.0f;
        for (unsigned int j = 0; j < count; ++j) {
            weight[j] = expf(weight[j] - best);
            sum += weight[j];
        }
        for (unsigned int j = 0; j < count; ++j) weight[j] /= sum;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float total = 0.0f;
        for (unsigned int j = 0; j < count; ++j) total += weight[j] * mine[(size_t)j * width + i];
        o[i] = total;
    }
}

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
