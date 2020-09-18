#pragma once

//================================================================
//
// float32
// float64
//
//================================================================

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__CUDA_ARCH__) || defined(__arm__) || defined(__aarch64__)

    using float32 = float;
    static_assert(sizeof(float32) == 4, "");
    static_assert(alignof(float32) == 4, "");

    using float64 = double;
    static_assert(sizeof(float64) == 8, "");
    static_assert(alignof(float64) == 8, "");

#else

    #error Check the platform and define the types

#endif
