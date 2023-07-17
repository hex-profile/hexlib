#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// debugBreak
//
//================================================================

void debugBreak();

//================================================================
//
// DEBUG_BREAK_INLINE
//
//================================================================

#if defined(__CUDA_ARCH__) || defined(__arm__) || defined(__aarch64__) || defined(__GNUC__)

    #define DEBUG_BREAK_INLINE() \
        void(0)

#elif defined(_MSC_VER)

    __forceinline void debugBreakInlineFunc()
        {__try {__debugbreak();} __except (1) {}}

    #define DEBUG_BREAK_INLINE() \
        debugBreakInlineFunc()

#else

    #error Implement

#endif

//================================================================
//
// DEBUG_BREAK_CHECK
//
//================================================================

#define DEBUG_BREAK_CHECK(condition) \
    (allv(condition) || (DEBUG_BREAK_INLINE(), false))

#define DEBUG_BREAK_IF(condition) \
    (!allv(condition) || (DEBUG_BREAK_INLINE(), false))
