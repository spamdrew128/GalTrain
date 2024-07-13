#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <stdlib.h>

void AddVecs(float *dest, const float *a, const float* b, size_t len);
void SqMatMul(int32_t *d_dest, const int32_t *d_a, const int32_t* d_b, size_t n);

#endif