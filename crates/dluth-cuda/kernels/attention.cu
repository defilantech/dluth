// The attention sublayer's operations other than its matrix products, for one token.
// Each mirrors one function of the CPU reference in dluth-kernels and adds a row's
// products in the same order, so the two differ only where the device rounds a multiply
// and an add together.
//
// Keys and values of earlier tokens live on the device in a history of `cap` rows. The
// token at position p is in row p % cap. A layer that sees everything has cap at least
// the longest sequence; a layer with a window has cap equal to the window, so a row is
// overwritten exactly when its token leaves the window.

// Each head's vector: multiplied element by element by `scale` (if not null), shortened
// to length `clip` if it is longer, then multiplied by `late`. One block of threads per
// head, a thread per few elements: a single thread walking a head of 128 took 25
// microseconds, more than the matrix product that made the head.
// CPU reference: the scale loop, cpu::l2_clip_inplace and the late factor in the model's
// attention step. The squares are added in a different order.
extern "C" __global__ void head_scale_clip(float* x, const float* scale, unsigned int heads,
                                           unsigned int dim, float clip, float late) {
    __shared__ float part[256];
    unsigned int h = blockIdx.x;
    if (h >= heads) return;
    float* v = x + (size_t)h * dim;
    float squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float t = scale ? v[i] * scale[(size_t)h * dim + i] : v[i];
        v[i] = t;
        squares += t * t;
    }
    part[threadIdx.x] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) squares += part[t];
        part[0] = fminf(clip / (sqrtf(squares) + 1e-6f), 1.0f);
    }
    __syncthreads();
    float factor = part[0];
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float t = v[i];
        if (factor < 1.0f) t *= factor;
        v[i] = t * late;
    }
}

// Each head's vector divided by sqrt(mean of its squares + eps). One block of threads per
// head. CPU reference: cpu::rmsnorm_plain, once per head, with the squares added in a
// different order.
extern "C" __global__ void head_rmsnorm_plain(const float* x, float* out, unsigned int heads,
                                              unsigned int dim, float eps) {
    __shared__ float part[256];
    unsigned int h = blockIdx.x;
    if (h >= heads) return;
    const float* v = x + (size_t)h * dim;
    float squares = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) squares += v[i] * v[i];
    part[threadIdx.x] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) squares += part[t];
        part[0] = 1.0f / sqrtf(squares / (float)dim + eps);
    }
    __syncthreads();
    float scale = part[0];
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) out[(size_t)h * dim + i] = v[i] * scale;
}

// The token `lag` steps before position `pos`, channel c, from a ring of the last n - 1
// tokens; zero before the start of the sequence.
__device__ __forceinline__ float earlier_token(const float* ring, unsigned int channels, unsigned int n,
                                               unsigned long long pos, unsigned int lag, unsigned int c) {
    return pos >= lag ? ring[((pos - lag) % (n - 1)) * channels + c] : 0.0f;
}

// Short convolution with fixed taps: taps[c * n + j] weighs the token n - 1 - j steps back.
//     out[c] = (sum of taps times tokens + current[c]) * norm
// Then the current token takes its place in the ring. CPU reference: sconv::fixed.
extern "C" __global__ void sconv_fixed(const float* current, float* ring, const float* taps, float* out,
                                       unsigned int channels, unsigned int n, unsigned long long pos,
                                       float norm) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float x = current[c];
    const float* w = taps + (size_t)c * n;
    float sum = w[n - 1] * x;
    for (unsigned int j = 0; j + 1 < n; ++j) sum += w[j] * earlier_token(ring, channels, n, pos, n - 1 - j, c);
    out[c] = (sum + x) * norm;
    if (n > 1) ring[(pos % (n - 1)) * channels + c] = x;
}

// Short convolution whose taps are a fixed part per channel plus a part computed for this
// token and shared by the channels of a head: fixed_taps[lag * channels + c] and
// token_taps[head * n + lag], lag 0 being the current token. CPU reference: sconv::dynamic.
extern "C" __global__ void sconv_dynamic(const float* current, float* ring, const float* fixed_taps,
                                         const float* token_taps, float* out, unsigned int channels,
                                         unsigned int n, unsigned int heads, unsigned long long pos,
                                         float norm) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float x = current[c];
    const float* head_taps = token_taps + (size_t)(c / (channels / heads)) * n;
    float sum = (fixed_taps[c] + head_taps[0]) * x;
    for (unsigned int lag = 1; lag < n; ++lag)
        sum += (fixed_taps[(size_t)lag * channels + c] + head_taps[lag]) * earlier_token(ring, channels, n, pos, lag, c);
    out[c] = (sum + x) * norm;
    if (n > 1) ring[(pos % (n - 1)) * channels + c] = x;
}

// Attention scores of one query token. For query head h and the token d steps back
// (d < seen): the dot product of the head's query and that token's key times `scale`,
// plus the relative-position term, sum over k of rel[h][k] * table[k][d], which is zero
// from distance `reach` on. weights[h * (seen + 1) + d] receives it; the last entry of a
// head, at d = seen, receives the head's sink score.
// CPU reference: the scoring loop of sinkattn::attend, with sinkattn::distance_profile.
extern "C" __global__ void attn_scores(const float* q, const float* keys, const float* rel,
                                       const float* table, const float* sinks, float* weights,
                                       unsigned int heads, unsigned int kv_heads, unsigned int dim,
                                       unsigned int rel_dim, unsigned int reach, float scale,
                                       unsigned long long pos, unsigned long long seen,
                                       unsigned long long cap) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long h = index / (seen + 1);
    unsigned long long d = index % (seen + 1);
    if (h >= heads) return;
    if (d == seen) {
        weights[index] = sinks[h];
        return;
    }
    const float* key = keys + ((pos - d) % cap) * kv_heads * dim + (h / (heads / kv_heads)) * dim;
    const float* query = q + h * dim;
    float dot = 0.0f;
    for (unsigned int i = 0; i < dim; ++i) dot += query[i] * key[i];
    float bias = 0.0f;
    if (d < reach)
        for (unsigned int k = 0; k < rel_dim; ++k) bias += rel[h * rel_dim + k] * table[(size_t)k * reach + d];
    weights[index] = dot * scale + bias;
}

// Softmax over each head's `count` scores, in place: one block of threads per head.
// CPU reference: cpu::softmax_inplace. The sum is added in a different order.
extern "C" __global__ void attn_softmax(float* weights, unsigned int heads, unsigned long long count) {
    __shared__ float part[1024];
    unsigned int h = blockIdx.x;
    if (h >= heads) return;
    float* w = weights + (unsigned long long)h * count;
    float best = -INFINITY;
    for (unsigned long long i = threadIdx.x; i < count; i += blockDim.x) best = fmaxf(best, w[i]);
    part[threadIdx.x] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) best = fmaxf(best, part[t]);
        part[0] = best;
    }
    __syncthreads();
    best = part[0];
    __syncthreads();
    float sum = 0.0f;
    for (unsigned long long i = threadIdx.x; i < count; i += blockDim.x) {
        float e = expf(w[i] - best);
        w[i] = e;
        sum += e;
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int t = 1; t < blockDim.x; ++t) sum += part[t];
        part[0] = sum;
    }
    __syncthreads();
    sum = part[0];
    for (unsigned long long i = threadIdx.x; i < count; i += blockDim.x) w[i] /= sum;
}

// The weighted sum of the values: out[h][e] = sum over the seen tokens, oldest first, of
// the token's weight times its value. The sink has no value.
// CPU reference: the summing loop of sinkattn::attend.
extern "C" __global__ void attn_values(const float* weights, const float* values, float* out,
                                       unsigned int heads, unsigned int kv_heads, unsigned int dim,
                                       unsigned long long pos, unsigned long long seen,
                                       unsigned long long cap) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long h = index / dim;
    unsigned int e = (unsigned int)(index % dim);
    if (h >= heads) return;
    const float* w = weights + h * (seen + 1);
    unsigned long long at = (h / (heads / kv_heads)) * dim + e;
    float total = 0.0f;
    for (unsigned long long d = seen; d-- > 0;) total += w[d] * values[((pos - d) % cap) * kv_heads * dim + at];
    out[index] = total;
}

// x[i] *= sigmoid(gate[i]). CPU reference: cpu::sigmoid.
extern "C" __global__ void sigmoid_mul(float* x, const float* gate, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
