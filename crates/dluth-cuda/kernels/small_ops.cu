// Small operations on whole vectors. None reads many bytes; they exist so that a token's
// values can stay on the device from one matrix product to the next.

// out[i] = a[i] + b[i].
extern "C" __global__ void vec_add(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// Root-mean-square normalization with learned weights, one block of threads:
//     out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
// CPU reference: cpu::rmsnorm. The squares are added in a different order.
extern "C" __global__ void rmsnorm_weighted(const float* x, const float* w, float* out, unsigned int n,
                                            float eps) {
    __shared__ float part[1024];
    float squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) squares += x[i] * x[i];
    part[threadIdx.x] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) squares += part[t];
        part[0] = squares;
    }
    __syncthreads();
    float scale = 1.0f / sqrtf(part[0] / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * w[i];
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
