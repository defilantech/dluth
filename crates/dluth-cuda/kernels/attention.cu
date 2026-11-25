// The attention sublayer's operations other than its matrix products, for one token.
// Each mirrors one function of the CPU reference in dluth-kernels and adds a row's
// products in the same order, so the two differ only where the device rounds a multiply
// and an add together.
//
// Keys and values of earlier tokens live on the device in a history of `cap` rows. The
// token at position p is in row p % cap. A layer that sees everything has cap at least
// the longest sequence; a layer with a window has cap equal to the window, so a row is
// overwritten exactly when its token leaves the window.

// How many tokens, the newest included, the token at position `pos` may look at: all of
// them, or the last `window` when the layer has a window (window 0 means none).
__device__ __forceinline__ unsigned long long tokens_seen(unsigned long long pos, unsigned long long window) {
    return (window != 0 && pos + 1 > window) ? window : pos + 1;
}

// Each head's vector: multiplied element by element by `scale` (if not null), shortened
// to length `clip` if it is longer, then multiplied by a factor that grows late in a long
// context: 1 + alpha * ln(max((position + 1) / floor, 1)), which is exactly 1 when alpha
// is 0. `x` holds `vectors` vectors of `dim`: for each position of a batch, `heads` of
// them; the scale repeats per position, and the first position is `pos`. One block of
// threads per vector.
// CPU reference: the scale loop, cpu::l2_clip_inplace and cpu::log_scale in the model's
// attention step. The squares are added in a different order.
extern "C" __global__ void head_scale_clip(float* x, const float* scale, unsigned int vectors,
                                           unsigned int heads, unsigned int dim, float clip,
                                           unsigned long long pos, float floor, float alpha) {
    __shared__ float part[256];
    unsigned int index = blockIdx.x;
    if (index >= vectors) return;
    unsigned int h = index % heads;
    unsigned long long at = pos + index / heads;
    float* v = x + (size_t)index * dim;
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
    float late = 1.0f + alpha * logf(fmaxf((float)(at + 1) / floor, 1.0f));
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float t = v[i];
        if (factor < 1.0f) t *= factor;
        v[i] = t * late;
    }
}

// Each head's vector divided by sqrt(mean of its squares + eps). One block of threads per
// head; for a batch, `heads` counts the heads of every position. CPU reference:
// cpu::rmsnorm_plain, once per head, with the squares added in a different order.
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

// The token `lag` steps before position b of a batch whose first token is at `pos`,
// channel c: from the batch itself if it is that recent, else from a ring of the last
// n - 1 tokens before the batch; zero before the start of the sequence.
__device__ __forceinline__ float earlier_token(const float* current, const float* ring, unsigned int channels,
                                               unsigned int n, unsigned long long pos, unsigned int b,
                                               unsigned int lag, unsigned int c) {
    if (b >= lag) return current[(size_t)(b - lag) * channels + c];
    unsigned long long at = pos + b;
    return at >= lag ? ring[((at - lag) % (n - 1)) * channels + c] : 0.0f;
}

// Short convolution with fixed taps, for a batch of `batch` consecutive tokens of
// `channels` floats: taps[c * n + j] weighs the token n - 1 - j steps back.
//     out[b][c] = (sum of taps times tokens + current[b][c]) * norm
// With a batch of one, the token then takes its place in the ring. With a larger batch
// the ring is only read here (another position's thread may be reading the slot a write
// would hit); `ring_store` puts the batch's last tokens in afterwards.
// CPU reference: sconv::fixed, token after token.
extern "C" __global__ void sconv_fixed(const float* current, float* ring, const float* taps, float* out,
                                       unsigned int channels, unsigned int n, unsigned long long pos,
                                       float norm, unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = (unsigned int)(index / channels);
    unsigned int c = (unsigned int)(index % channels);
    if (b >= batch) return;
    float x = current[index];
    const float* w = taps + (size_t)c * n;
    float sum = w[n - 1] * x;
    for (unsigned int j = 0; j + 1 < n; ++j) sum += w[j] * earlier_token(current, ring, channels, n, pos, b, n - 1 - j, c);
    out[index] = (sum + x) * norm;
    if (batch == 1 && n > 1) ring[(pos % (n - 1)) * channels + c] = x;
}

// Short convolution whose taps are a fixed part per channel plus a part computed for each
// token and shared by the channels of a head: fixed_taps[lag * channels + c] and
// token_taps[b][head * n + lag], lag 0 being the current token. The ring as in
// sconv_fixed. CPU reference: sconv::dynamic, token after token.
extern "C" __global__ void sconv_dynamic(const float* current, float* ring, const float* fixed_taps,
                                         const float* token_taps, float* out, unsigned int channels,
                                         unsigned int n, unsigned int heads, unsigned long long pos,
                                         float norm, unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = (unsigned int)(index / channels);
    unsigned int c = (unsigned int)(index % channels);
    if (b >= batch) return;
    float x = current[index];
    const float* head_taps = token_taps + ((size_t)b * heads + c / (channels / heads)) * n;
    float sum = (fixed_taps[c] + head_taps[0]) * x;
    for (unsigned int lag = 1; lag < n; ++lag)
        sum += (fixed_taps[(size_t)lag * channels + c] + head_taps[lag])
               * earlier_token(current, ring, channels, n, pos, b, lag, c);
    out[index] = (sum + x) * norm;
    if (batch == 1 && n > 1) ring[(pos % (n - 1)) * channels + c] = x;
}

// After a convolution over a batch larger than one: the batch's last n - 1 tokens (or all
// of them, if it has fewer) take their places in the ring. The token at position p goes to
// slot p % (n - 1); the tokens stored here have different slots.
extern "C" __global__ void ring_store(const float* current, float* ring, unsigned int channels,
                                      unsigned int n, unsigned long long pos, unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = (unsigned int)(index / channels);
    unsigned int c = (unsigned int)(index % channels);
    if (b >= batch || n < 2) return;
    if (batch - b > n - 1) return;
    ring[((pos + b) % (n - 1)) * channels + c] = current[index];
}

// The keys or values of a batch go to their rows of the history: the token at position p
// is in row p % cap. The rows of a batch may wrap around the end of the history.
extern "C" __global__ void store_rows(const float* rows, float* history, unsigned int width,
                                      unsigned long long cap, unsigned long long pos, unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long b = index / width;
    if (b >= batch) return;
    history[((pos + b) % cap) * width + index % width] = rows[index];
}

// Attention scores of a batch of query tokens, the first at position `pos`. The query at
// position p sees tokens_seen(p, window) tokens. For query b, head h and the token d steps
// back (d < seen): the dot product of the head's query and that token's key times `scale`,
// plus the relative-position term, sum over k of rel[b][h][k] * table[k][d], which is zero
// from distance `reach` on. Each (query, head) has a run of `stride` floats in `weights`:
// entry d receives the score, entry `seen` the head's sink score, and the rest is unused.
// CPU reference: the scoring loop of sinkattn::attend, with sinkattn::distance_profile.
extern "C" __global__ void attn_scores(const float* q, const float* keys, const float* rel,
                                       const float* table, const float* sinks, float* weights,
                                       unsigned int heads, unsigned int kv_heads, unsigned int dim,
                                       unsigned int rel_dim, unsigned int reach, float scale,
                                       unsigned long long pos, unsigned long long window,
                                       unsigned long long cap, unsigned long long stride,
                                       unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long run = index / stride;
    unsigned long long d = index % stride;
    if (run >= (unsigned long long)batch * heads) return;
    unsigned long long b = run / heads;
    unsigned int h = (unsigned int)(run % heads);
    unsigned long long at = pos + b;
    unsigned long long seen = tokens_seen(at, window);
    if (d > seen) return;
    if (d == seen) {
        weights[index] = sinks[run];
        return;
    }
    const float* key = keys + ((at - d) % cap) * kv_heads * dim + (size_t)(h / (heads / kv_heads)) * dim;
    const float* query = q + run * dim;
    float dot = 0.0f;
    for (unsigned int i = 0; i < dim; ++i) dot += query[i] * key[i];
    float bias = 0.0f;
    if (d < reach)
        for (unsigned int k = 0; k < rel_dim; ++k) bias += rel[run * rel_dim + k] * table[(size_t)k * reach + d];
    weights[index] = dot * scale + bias;
}

// Softmax over each (query, head) run's scores and its sink, in place: one block of
// threads per run. CPU reference: cpu::softmax_inplace. The sum is added in a different
// order.
extern "C" __global__ void attn_softmax(float* weights, unsigned int heads, unsigned long long pos,
                                        unsigned long long window, unsigned long long stride,
                                        unsigned int batch) {
    __shared__ float part[1024];
    unsigned long long run = blockIdx.x;
    if (run >= (unsigned long long)batch * heads) return;
    unsigned long long count = tokens_seen(pos + run / heads, window) + 1;
    float* w = weights + run * stride;
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

// The weighted sum of the values: out[b][h][e] = sum over the tokens query b sees, oldest
// first, of the token's weight times its value. The sink has no value.
// CPU reference: the summing loop of sinkattn::attend.
extern "C" __global__ void attn_values(const float* weights, const float* values, float* out,
                                       unsigned int heads, unsigned int kv_heads, unsigned int dim,
                                       unsigned long long pos, unsigned long long window,
                                       unsigned long long cap, unsigned long long stride,
                                       unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long run = index / dim;
    unsigned int e = (unsigned int)(index % dim);
    if (run >= (unsigned long long)batch * heads) return;
    unsigned int h = (unsigned int)(run % heads);
    unsigned long long at = pos + run / heads;
    unsigned long long seen = tokens_seen(at, window);
    const float* w = weights + run * stride;
    unsigned long long column = (size_t)(h / (heads / kv_heads)) * dim + e;
    float total = 0.0f;
    for (unsigned long long d = seen; d-- > 0;) total += w[d] * values[((at - d) % cap) * kv_heads * dim + column];
    out[index] = total;
}

// x[i] *= sigmoid(gate[i]). CPU reference: cpu::sigmoid.
extern "C" __global__ void sigmoid_mul(float* x, const float* gate, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
