#pragma once

#include <stddef.h>

//================================================================
//
// HEXLIB_INLINE
//
//================================================================

#if defined(__CUDA_ARCH__)
    #define HEXLIB_INLINE __device__ __host__ inline
#elif defined(_MSC_VER)
    #define HEXLIB_INLINE __forceinline
#else
    #define HEXLIB_INLINE inline
#endif
