#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Builtin floating-point types
//
//================================================================

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__CUDA_ARCH__) || defined(__arm__) || defined(__aarch64__)

//----------------------------------------------------------------

#ifndef HEXLIB_FLOAT32_AND_FLOAT64
#define HEXLIB_FLOAT32_AND_FLOAT64

using float32 = float;
COMPILE_ASSERT(sizeof(float32) == 4 && alignof(float32) == 4);

using float64 = double;
COMPILE_ASSERT(sizeof(float64) == 8 && alignof(float64) == 8);

#endif

//----------------------------------------------------------------

#define BUILTIN_FLOAT_FOREACH_NORMAL(action, extra) \
    action(float32, extra)

#define BUILTIN_FLOAT_FOREACH_LARGE(action, extra) \
    action(float64, extra)

#define BUILTIN_FLOAT_FOREACH(action, extra) \
    BUILTIN_FLOAT_FOREACH_NORMAL(action, extra) \
    BUILTIN_FLOAT_FOREACH_LARGE(action, extra)

//----------------------------------------------------------------

#else

    #error Check the platform and define the types

#endif
