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

//================================================================
//
// HEXLIB_NODISCARD
//
//================================================================

#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
    #define HEXLIB_NODISCARD [[nodiscard]]
#else
    #define HEXLIB_NODISCARD
#endif

//================================================================
//
// HEXLIB_ENSURE
//
//================================================================

#define HEXLIB_ENSURE(condition) \
    if (condition) ; else return false
