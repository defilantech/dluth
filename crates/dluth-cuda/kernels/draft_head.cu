// The draft head's operations that are not matrix products or the model's own small
// operations. The head (a small network that guesses the next few tokens) is described
// where it is implemented on the CPU, dluth-model's inkling/mtp.rs; each kernel here
// names the part of that implementation it stands for.

// Each vector of `dim` in `x`: divided by sqrt(mean of its squares + eps), multiplied
// element by element by `weights` (one vector of `dim`, the same for all), then rotated by
// its position: element i and element i + dim/2 are a pair, turned by the angle
// position * theta^(-2i/dim). `x` holds, for each of `count` positions, `heads` vectors;
// the positions are `pos`, `pos + 1`, ... One block of threads per vector; dim is at most
// 256 and even.
// CPU reference: MtpHead::head_norm_rope (cpu::rmsnorm per head, then cpu::rope_inplace).
extern "C" __global__ void head_norm_rope(float* x, const float* weights, unsigned int count,
                                          unsigned int heads, unsigned int dim, float eps,
                                          unsigned long long pos, float theta) {
    __shared__ float part[256];
    __shared__ float normed[256];
    unsigned int vector = blockIdx.x;
    if (vector >= count * heads) return;
    float* v = x + (size_t)vector * dim;
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
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) normed[i] = v[i] * scale * weights[i];
    __syncthreads();
    unsigned int half = dim / 2;
    float at = (float)(pos + vector / heads);
    for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
        float frequency = 1.0f / powf(theta, (float)(2 * i) / (float)dim);
        float s, c;
        sincosf(at * frequency, &s, &c);
        float a = normed[i], b = normed[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

// The block's attention scores. A block of `block` positions follows `context` positions
// of text. For block position k, head h and key j: the dot product of the head's query
// with key j's vector for the head's key-value group, times `scale`. Keys 0 to
// context - 1 are the text's (`context_keys`, one row of kv_heads * dim per position);
// keys context to context + block - 1 are the block's own (`block_keys`): every block
// position sees the whole block. Each (position, head) has a run of context + block scores.
// A local layer is given only the text its anchor sees, and a block position further on
// sees less of it: position k does not see the text's keys j with j + slack < k, whose
// scores are minus infinity (a weight of zero after the softmax). A layer that sees the
// whole text passes a slack of at least block - 1.
// CPU reference: the scoring in MtpDrafter::block.
extern "C" __global__ void draft_scores(const float* q, const float* context_keys, const float* block_keys,
                                        float* scores, unsigned int heads, unsigned int kv_heads,
                                        unsigned int dim, unsigned long long context,
                                        unsigned int block, float scale, unsigned int slack) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long keys = context + block;
    unsigned long long run = index / keys;
    unsigned long long j = index % keys;
    if (run >= (unsigned long long)block * heads) return;
    unsigned int h = (unsigned int)(run % heads);
    size_t group = (size_t)(h / (heads / kv_heads)) * dim;
    const float* key = (j < context ? context_keys + j * kv_heads * dim : block_keys + (j - context) * kv_heads * dim) + group;
    const float* query = q + run * dim;
    float dot = 0.0f;
    for (unsigned int i = 0; i < dim; ++i) dot += query[i] * key[i];
    bool hidden = j < context && j + slack < run / heads;
    scores[index] = hidden ? -INFINITY : dot * scale;
}

// The weighted sum of the values for the block's attention: out[k][h][e] = sum over the
// keys j of weights[k][h][j] times value j's element e for the head's group. `segs`
// threads share one output element, each taking every segs-th key, combined with warp
// shuffles (segs a power of two up to 32, the block size a multiple of 32).
// CPU reference: the summing in MtpDrafter::block.
extern "C" __global__ void draft_values(const float* weights, const float* context_values,
                                        const float* block_values, float* out, unsigned int heads,
                                        unsigned int kv_heads, unsigned int dim,
                                        unsigned long long context, unsigned int block,
                                        unsigned int segs) {
    unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long element = thread / segs;
    unsigned int seg = (unsigned int)(thread % segs);
    unsigned long long run = element / dim;
    unsigned int e = (unsigned int)(element % dim);
    unsigned long long keys = context + block;
    bool live = run < (unsigned long long)block * heads;
    float total = 0.0f;
    if (live) {
        unsigned int h = (unsigned int)(run % heads);
        size_t column = (size_t)(h / (heads / kv_heads)) * dim + e;
        const float* w = weights + run * keys;
        for (unsigned long long j = seg; j < keys; j += segs) {
            const float* value = j < context ? context_values + j * kv_heads * dim : block_values + (j - context) * kv_heads * dim;
            total += w[j] * value[column];
        }
    }
    for (unsigned int offset = segs / 2; offset > 0; offset /= 2) total += __shfl_xor_sync(0xffffffffu, total, offset);
    if (live && seg == 0) out[element] = total;
}

// out[i] = gate[i] * sigmoid(gate[i]) * up[i]: the gated activation of the head's
// feed-forward part. CPU reference: cpu::silu_mul.
extern "C" __global__ void silu_mul(const float* gate, const float* up, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = g / (1.0f + expf(-g)) * up[i];
}

// out[i] = w0 * a[i] + w1 * b[i] + w2 * c[i] + w3 * d[i]: the mix of up to four of the
// model's values that a layer of the head reads its context from (unused ones are given a
// weight of 0 and any valid pointer). CPU reference: the sum in MtpHead::mixed.
extern "C" __global__ void mix_four(const float* a, const float* b, const float* c, const float* d,
                                    float* out, float w0, float w1, float w2, float w3, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w0 * a[i] + w1 * b[i] + w2 * c[i] + w3 * d[i];
}
