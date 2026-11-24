// Elementwise sum of two float arrays. The first Dlúth kernel: it exists to prove the
// path from source to an embedded, launchable GPU kernel, not to be useful by itself.
// CPU reference: dluth_kernels::cpu::vecadd.
extern "C" __global__ void vecadd_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
