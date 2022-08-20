#pragma once
#include "nn/nn_global.h"
#include "nn/tensor.h"

#ifdef NN_ENABLE_AVX_INTRINSICS
#include "immintrin.h"
float _avx_float32_dot(float* a, float* b, int n);
double _avx_float64_dot(double* a, double* b, int n);
int _avx_int32_dot(int* a, int* b, int n);
#endif
