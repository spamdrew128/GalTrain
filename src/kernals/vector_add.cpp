#include <hip/hip_runtime.h>
#include "kernal_constants.h"

#define H_A(x) assert((x) == hipSuccess)

__global__ void KernalVectorAdd(float *dest, const float *a, const float* b, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < len) {
        dest[id] = a[id] + b[id];
    }
}

extern "C" void AddVecs(float *d_dest, const float *d_a, const float* d_b, size_t len) {
    const size_t num_blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    KernalVectorAdd<<<num_blocks, threadsPerBlock>>>(d_dest, d_a, d_b, len);
}
