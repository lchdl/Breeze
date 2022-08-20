#include "nn/avx.h"

#ifdef NN_ENABLE_AVX_INTRINSICS
float _avx_float32_dot(float* a, float* b, int n) {
    const int _pack = 8;
    const int _groups = n / _pack;
    __m256 vsum = _mm256_setzero_ps();
    float f[8], dot = 0.0f;
    for (int i = 0; i < _pack * _groups; i += _pack) {
        __m256 A = _mm256_loadu_ps(&(a[i]));
        __m256 B = _mm256_loadu_ps(&(b[i]));
        __m256 C = _mm256_mul_ps(A, B);
        vsum = _mm256_add_ps(vsum, C);
    }
    _mm256_storeu_ps(&f[0], vsum);
    /* add remaining numbers */
    for (int i = _pack * _groups; i < n; i++)
        dot += a[i] * b[i];
    for (int i = 0; i < _pack; i++)
        dot += f[i];
    return dot;
}
double _avx_float64_dot(double* a, double* b, int n)
{
    const int _pack = 4;
    const int _groups = n / _pack;
    __m256d vsum = _mm256_setzero_pd();
    double f[8], dot = 0.0;
    for (int i = 0; i < _pack * _groups; i += _pack) {
        __m256d A = _mm256_loadu_pd(&(a[i]));
        __m256d B = _mm256_loadu_pd(&(b[i]));
        __m256d C = _mm256_mul_pd(A, B);
        vsum = _mm256_add_pd(vsum, C);
    }
    _mm256_storeu_pd(&f[0], vsum);
    /* add remaining numbers */
    for (int i = _pack * _groups; i < n; i++)
        dot += a[i] * b[i];
    for (int i = 0; i < _pack; i++)
        dot += f[i];
    return dot;
}
int _avx_int32_dot(int * a, int * b, int n)
{
    const int _pack = 8;
    const int _groups = n / _pack;
    __m256i vsum = _mm256_setzero_si256();
    int v[8], dot = 0;
    for (int i = 0; i < _pack * _groups; i += _pack) {
        __m256i A = _mm256_loadu_si256((__m256i*)&(a[i]));
        __m256i B = _mm256_loadu_si256((__m256i*)&(b[i]));
        __m256i C = _mm256_mul_epi32(A, B);
        vsum = _mm256_add_epi32(vsum, C);
    }
    _mm256_storeu_si256((__m256i*)&v[0], vsum);
    /* add remaining numbers */
    for (int i = _pack * _groups; i < n; i++)
        dot += a[i] * b[i];
    for (int i = 0; i < _pack; i++)
        dot += v[i];
    return dot;
}
#endif
