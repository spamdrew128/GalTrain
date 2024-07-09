#include <hip/hip_runtime.h>

#define H_A(x) assert((x) == hipSuccess)

__global__ void KernalVectorAdd(float *dest, const float *a, const float* b, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < len) {
        dest[id] = a[id] + b[id];
    }
}

void AddVecs(float *dest, const float *a, const float* b, size_t len) {
    size_t arr_bytes = len * sizeof(float);

    float *d_a, *d_b, *d_dest;
    H_A(hipMalloc((void**)&d_a, arr_bytes));
    H_A(hipMalloc((void**)&d_b, arr_bytes));
    H_A(hipMalloc((void**)&d_dest, arr_bytes));

    H_A(hipMemcpy(d_a, a, arr_bytes, hipMemcpyHostToDevice));
    H_A(hipMemcpy(d_b, b, arr_bytes, hipMemcpyHostToDevice));

    uint32_t block_size = 256;
    uint32_t num_blocks = (N + block_size - 1) / block_size;

    VectorAdd<<<num_blocks, block_size>>>(d_dest, d_a, d_b, len);
    H_A(hipDeviceSynchronize());

    H_A(hipMemcpy(dest, d_dest, arr_bytes, hipMemcpyDeviceToHost));

    float *cpu_dest = (float*)calloc(N, sizeof(*cpu_dest));
    CpuVectorAdd(cpu_dest, a, b);
    assert(VecCmp(dest, cpu_dest));

    H_A(hipFree(d_a));
    H_A(hipFree(d_b));
    H_A(hipFree(d_dest));
}