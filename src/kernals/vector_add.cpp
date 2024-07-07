#include <hip/hip_runtime.h>

#define H_A(x) assert((x) == hipSuccess)

__global__ void VectorAdd(float *dest, const float *a, const float* b, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < len) {
        dest[id] = a[id] + b[id];
    }
}

void AddVecs(float *dest, const float *a, const float* b, size_t len) {
    float *d_a, *d_b, *d_dest;
    H_A(hipMalloc((void**)&d_a, len));
    H_A(hipMalloc((void**)&d_b, len));
    H_A(hipMalloc((void**)&d_dest, len));
 
    H_A(hipMemcpy(d_a, a, len, hipMemcpyHostToDevice));
    H_A(hipMemcpy(d_b, b, len, hipMemcpyHostToDevice));

    size_t blockSize = 256;
    size_t blockCount = (len + blockSize - 1) / blockSize;

    VectorAdd<<<blockCount, blockSize>>>(d_dest, d_a, d_b, len);
    H_A(hipDeviceSynchronize());

    H_A(hipMemcpy(dest, d_dest, len, hipMemcpyDeviceToHost));

    H_A(hipFree(d_a));
    H_A(hipFree(d_b));
    H_A(hipFree(d_dest));
}