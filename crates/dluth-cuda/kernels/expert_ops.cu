// The two small steps of an expert layer that sit between and after its matrix products.
// They are cheap arithmetic, but on the CPU they cost a round trip to the device and well
// over a million tanh and exp calls per token; here they keep the whole expert stage on
// the device.

// The activation inside an expert, for a run of (gate, up) pairs:
//     act[i] = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
// where gate = both[2i], up = both[2i + 1] and softcap(x, c) = c * tanh(x / c).
// The sigmoid sees the raw gate, not the limited one.
// CPU reference: dluth_kernels::cpu::capped_gate_mul.
extern "C" __global__ void capped_gate_pairs(const float* both, float* act, unsigned int pairs,
                                             float gate_cap, float up_cap) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    float2 gu = reinterpret_cast<const float2*>(both)[i];
    float gate = gate_cap * tanhf(gu.x / gate_cap);
    float up = up_cap * tanhf(gu.y / up_cap);
    act[i] = gate * (1.0f / (1.0f + expf(-gu.x))) * up;
}

// The weighted sum of the chosen experts' outputs, for a batch of positions:
//     y[b][r] = sum over j of weights[b][j] * outs[b][j][r]
// `outs` holds, for each position, `count` vectors of `width` floats, one after another;
// `weights` holds `count` floats per position. The sum runs in the order of j, like the
// CPU reference (the loop in the model's expert step).
extern "C" __global__ void weighted_sum(const float* outs, const float* weights, float* y,
                                        unsigned int count, unsigned int width, unsigned int batch) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long b = index / width;
    unsigned int r = (unsigned int)(index % width);
    if (b >= batch) return;
    const float* mine = outs + b * count * width;
    const float* w = weights + b * count;
    float total = 0.0f;
    for (unsigned int j = 0; j < count; ++j) total += w[j] * mine[(size_t)j * width + r];
    y[index] = total;
}
