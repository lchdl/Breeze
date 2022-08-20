#pragma once

/* global compile settings */
#define NN_ENABLE_RUNTIME_CHECKING /* enable parameter check during runtime */
#define NN_DEBUG_BREAK_WHEN_ERROR  /* determine if a call to T_error(...) will trigger debug break in MSVC or GCC */
#define NN_ENABLE_AVX_INTRINSICS   /* enable AVX support (if available)     */
#define NN_ENABLE_OPENMP           /* enable OpenMP to speed up calculation */
