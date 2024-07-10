#include <hip/hip_runtime.h>
#include "kernal_constants.h"

#define H_A(x) assert((x) == hipSuccess)

__global__ void KernalVectorAdd(float *dest, const float *a, const float* b, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < len) {
        dest[id] = a[id] + b[id];
    }
}

extern "C" void AddVecs(float *dest, const float *a, const float* b, size_t len) {
    size_t arrBytes = len * sizeof(float);

    float *d_a, *d_b, *d_dest;
    H_A(hipMalloc((void**)&d_a, arrBytes));
    H_A(hipMalloc((void**)&d_b, arrBytes));
    H_A(hipMalloc((void**)&d_dest, arrBytes));

    H_A(hipMemcpy(d_a, a, arrBytes, hipMemcpyHostToDevice));
    H_A(hipMemcpy(d_b, b, arrBytes, hipMemcpyHostToDevice));

    uint32_t num_blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    KernalVectorAdd<<<num_blocks, threadsPerBlock>>>(d_dest, d_a, d_b, len);

    H_A(hipDeviceSynchronize());

    H_A(hipMemcpy(dest, d_dest, arrBytes, hipMemcpyDeviceToHost));

    H_A(hipFree(d_a));
    H_A(hipFree(d_b));
    H_A(hipFree(d_dest));
}