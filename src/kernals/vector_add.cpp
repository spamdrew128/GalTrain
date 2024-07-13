#include <hip/hip_runtime.h>
#include <cmath>
#include "kernal_constants.h"

#define H_A(x) assert((x) == hipSuccess)

__global__ void K_VectorAdd(float *dest, const float *a, const float *b, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < len) {
        dest[id] = a[id] + b[id];
    }
}

extern "C" void AddVecs(float *d_dest, const float *d_a, const float *d_b, size_t len) {
    const size_t num_blocks = (len + ThreadsPerBlock - 1) / ThreadsPerBlock;
    K_VectorAdd<<<num_blocks, ThreadsPerBlock>>>(d_dest, d_a, d_b, len);
}

__global__ void K_SqMatMul(int32_t *dest, const int32_t *a, const int32_t *b, size_t n) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t dotProduct = 0;
    if((col < n) && (row < n)) {
        for(size_t i = 0; i < n; i++) {
            // mat[r][c] = mat[r*n + c]
            dotProduct += a[row*n + i] * b[i*n + col];
        }

        dest[row*n + col] = dotProduct;
    }
}

extern "C" void SqMatMul(int32_t *d_dest, const int32_t *d_a, const int32_t *d_b, size_t n) {
    size_t blockDimX = sqrt(ThreadsPerBlock);
    size_t gridDimX = (n + blockDimX - 1) / blockDimX;
    dim3 dimBlock(blockDimX, blockDimX);
    dim3 dimGrid(gridDimX, gridDimX);

    K_SqMatMul<<<dimGrid, dimBlock>>>(d_dest, d_a, d_b, n);
}