#pragma once

#include "compileTools/msvcTools.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Generic parallel reduction algorithm, without memory storage aspects.
//
// On each iteration, each ith thread accumulates a value:
//
// if (i < activeCount) data[i] += data[stageSize + i]
//
// Usually ptr = &data[i] is precalculated within the thread's state,
// so the operation looks like:
//
// if (i < activeCount) ptr[0] += ptr[stageSize]
//
// The reduction range is [0, reductionSize - 1]
// The reduction thread index is named "reductionMember".
//
// The number of activeFlag threads:
//
// stageSize + i < reductionSize ==>
// i < reductionSize - stageSize
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#define PARALLEL_REDUCTION_TWICE(reductionSize, reductionMember, actionMacro, actionContext) \
    MSVC_EXCLUDE(COMPILE_ASSERT((reductionSize) <= 1024)); \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 512) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 256) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 128) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 64) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 32) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 16) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 8) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 4) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 2) \
    PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, 1)

//----------------------------------------------------------------

#define PARALLEL_REDUCTION_TWICE_ITER(reductionSize, reductionMember, actionMacro, actionContext, stageSize) \
    { \
        MSVC_SELECT(const, constexpr) Space _activeCount = COMPILE_CLAMP((reductionSize) - (stageSize), 0, stageSize); \
        \
        if (_activeCount != 0) /* Removes code for unused iterations at compile time. */ \
        { \
            bool _isActive = (reductionMember) < _activeCount; \
            actionMacro(_isActive, stageSize, actionContext) \
        } \
    }
