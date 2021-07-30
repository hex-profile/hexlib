#pragma once

#include "gpuSupport/parallelReduction.h"
#include "prepTools/prepSeqForeach.h"

//================================================================
//
// The idea is:
//
// * Each warp reduce value in registers using modern devShuffle* intrinsics.
//
// * Each warp 0th lane saves the result to SRAM array
// (devWarpSize=32 times less elements than the thread count).
// 
// * The main 0th warp reads all valid results.
// * The main warp reduces value finally, also in registers.
//
//================================================================


//================================================================
//
// REDUCEMODERN__PREPARE
//
// Allocates SRAM arrays for all enumerated parameters.
// Each setup has a pre and may be used multiple times.
//
//================================================================

#define REDUCEMODERN__PREPARE(pre, threadCount, threadIndex, paramSeq) \
    \
    constexpr Space pre##ThreadCount = (threadCount); \
    const Space pre##ThreadIndex = (threadIndex); \
    \
    COMPILE_ASSERT(pre##ThreadCount % devWarpSize == 0); \
    constexpr Space pre##WarpCount = pre##ThreadCount / devWarpSize; \
    \
    const Space pre##WarpIndex = SpaceU(pre##ThreadIndex) / SpaceU(devWarpSize); \
    const Space pre##LaneIndex = SpaceU(pre##ThreadIndex) % SpaceU(devWarpSize); \
    \
    REDUCEMODERN__SETUP(pre, paramSeq)

//----------------------------------------------------------------

#define REDUCEMODERN__SETUP(pre, paramSeq) \
    PREP_SEQ_FOREACH_TRIPLET(paramSeq, REDUCEMODERN__SETUP_ITER, pre)

#define REDUCEMODERN__SETUP_ITER(Type, name, neutralValue, pre) \
    devSramArray(pre##name##FinalReduction, Type, pre##WarpCount);

//================================================================
//
// REDUCEMODERN__ITERATION
//
//================================================================

#define REDUCEMODERN__ITERATION(pre, paramSeq, body, shift) \
    \
    { \
        PREP_SEQ_FOREACH_TRIPLET(paramSeq, REDUCEMODERN__ITERATION_VARS, shift) \
        constexpr bool active = true; \
        {body;} \
    }

#define REDUCEMODERN__ITERATION_VARS(Type, name, neutralValue, shift) \
    auto* name##L = &name; \
    auto name##_Incoming = devShuffleDown(name, shift); \
    const auto* name##R = &name##_Incoming;

//================================================================
//
// REDUCEMODERN__STORE
//
// Store values to the final reduction array.
//
//================================================================

#define REDUCEMODERN__STORE(pre, paramSeq) \
    { \
        PREP_SEQ_FOREACH_TRIPLET(paramSeq, REDUCEMODERN__STORE_ITER, pre) \
    }

#define REDUCEMODERN__STORE_ITER(Type, name, neutralValue, pre) \
    pre##name##FinalReduction[pre##WarpIndex] = name;

//================================================================
//
// REDUCEMODERN__LOAD
//
//================================================================

#define REDUCEMODERN__LOAD(pre, paramSeq) \
    { \
        PREP_SEQ_FOREACH_TRIPLET(paramSeq, REDUCEMODERN__LOAD_ITER, pre) \
    }

#define REDUCEMODERN__LOAD_ITER(Type, name, neutralValue, pre) \
    name = (pre##LaneIndex < pre##WarpCount) ? pre##name##FinalReduction[pre##LaneIndex] : (neutralValue);

//================================================================
//
// REDUCEMODERN__APPLY
//
//================================================================

#define REDUCEMODERN__APPLY(pre, paramSeq, body) \
    { \
        /* Reduce values inside the current warp */ \
        COMPILE_ASSERT(devWarpSize == 32); \
        REDUCEMODERN__ITERATION(pre, paramSeq, body, 16) \
        REDUCEMODERN__ITERATION(pre, paramSeq, body, 8) \
        REDUCEMODERN__ITERATION(pre, paramSeq, body, 4) \
        REDUCEMODERN__ITERATION(pre, paramSeq, body, 2) \
        REDUCEMODERN__ITERATION(pre, paramSeq, body, 1) \
        \
        if (pre##WarpCount > 1) \
        { \
            /* Finish previous reading of the memory as the same SRAM context may be used multiple times. */ \
            /* Hoping that placing the two syncthreads together allows warps to be freer between reductions. */ \
            devSyncThreads(); \
            \
            /* Zeroth lane of each warp stores the warp result. */ \
            if (pre##LaneIndex == 0) \
                REDUCEMODERN__STORE(pre, paramSeq); \
            \
            devSyncThreads(); \
            \
            /* Load valid values back to registers. */ \
            COMPILE_ASSERT(pre##WarpCount <= devWarpSize); \
            REDUCEMODERN__LOAD(pre, paramSeq); \
            \
            /* Final register reduction, 0th thread will contain the result. */ \
            \
            COMPILE_ASSERT(devWarpSize == 32); \
            \
            if (pre##WarpCount > 16) \
                REDUCEMODERN__ITERATION(pre, paramSeq, body, 16) \
            \
            if (pre##WarpCount > 8) \
                REDUCEMODERN__ITERATION(pre, paramSeq, body, 8) \
            \
            if (pre##WarpCount > 4) \
                REDUCEMODERN__ITERATION(pre, paramSeq, body, 4) \
            \
            if (pre##WarpCount > 2) \
                REDUCEMODERN__ITERATION(pre, paramSeq, body, 2) \
            \
            if (pre##WarpCount > 1) \
                REDUCEMODERN__ITERATION(pre, paramSeq, body, 1) \
        } \
    }

//================================================================
//
// REDUCEMODERN__TRANSLATE
//
// Translates parameter sequence to the classic format
// (without neutral value).
//
//================================================================

#define REDUCEMODERN__TRANSLATE(paramSeq) \
    PREP_SEQ_FOREACH_TRIPLET(paramSeq, REDUCEMODERN__TRANSLATE_ITER0, _)

#define REDUCEMODERN__TRANSLATE_ITER0(Type, name, neutralValue, _) \
    ((Type, name))

//================================================================
//
// API
//
//================================================================

#if defined(__CUDA_ARCH__)

    #define REDUCTION_MODERN_PREPARE(pre, threadCount, threadIndex, paramSeq) \
        REDUCEMODERN__PREPARE(pre, threadCount, threadIndex, paramSeq)

    #define REDUCTION_MODERN_APPLY(pre, paramSeq, body) \
        REDUCEMODERN__APPLY(pre, paramSeq, body)

#else

    #define REDUCTION_MODERN_PREPARE(pre, threadCount, threadIndex, paramSeq) \
        REDUCTION_CLASSIC_PREPARE(pre, threadCount, threadIndex, REDUCEMODERN__TRANSLATE(paramSeq))

    #define REDUCTION_MODERN_APPLY(pre, paramSeq, body) \
        REDUCTION_CLASSIC_APPLY(pre, REDUCEMODERN__TRANSLATE(paramSeq), body)

#endif

//----------------------------------------------------------------

#define REDUCTION_MODERN_MAKE(pre, threadCount, threadIndex, paramSeq, body) \
    { \
        REDUCTION_MODERN_PREPARE(pre, threadCount, threadIndex, paramSeq) \
        REDUCTION_MODERN_APPLY(pre, paramSeq, body) \
    }
