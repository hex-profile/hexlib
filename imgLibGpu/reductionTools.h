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
// Usually ptr = &data[i] is precalculated inside the thread's state, 
// so the operation looks like:
//
// if (i < activeCount) ptr[0] += ptr[stageSize]
//
// The reduction range is [0, reductionSize - 1]
// The reduction thread index is named "reductionMember".
//
// The number of activeFlag threads:
// stageSize + i < reductionSize <===> i < reductionSize - stageSize
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

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Convenient tools to make reduction.
//
//----------------------------------------------------------------
//
// The tools support arbitrary number of parameters, 
// each parameter is given by pair (Type, name).
//
// All parameters are passed as a preprocessor sequence (a) (b) (c), for example:
// ((int, myInt)) ((float, myFloat).
//
//----------------------------------------------------------------
//
// The tools support N simultaneous parallel reductions:
//
// * outerSize is the number of reductions.
// * outerMember is the current reduction index.
//
//----------------------------------------------------------------
//
// The usage example:
//
// REDUCTION_MAKE
// (
//     myName,
//     tileSize, tileMember, // The number of reductions and the reduction index
//     cellSize, cellMember, // The reduction size and the index inside the reduction
//
//     // List of pairs {Type, name}
//     ((float32, sumWeight)) 
//     ((float32_x2, sumWeightValue))
//     ((float32_x2, sumWeightValueSq)),
//
//     {
//         // Inside the body, each computation checks "active" variable.
//         // (It checks "active" multiple times as it is efficient on GPU, predicated instructions)
//         // L and R suffixes mean lvalue and rvalue.
//
//         if (active) *sumWeightL += *sumWeightR;
//         if (active) *sumWeightValueL += *sumWeightValueR;
//         if (active) *sumWeightValueSqL += *sumWeightValueSqR;
//     }
// )
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================


//================================================================
//
// REDUCTION_PREPARE
//
// Allocates SRAM arrays for all enumerated parameters, 
// each array for N=outerSize simultaneous reductions.
//
// Flat size and flat member are total ones for the flattened arrays.
//
// Each setup has a prefix and may be used multiple times.
//
//================================================================

#define REDUCTION_PREPARE(prefix, outerSize, outerMember, reductionSize, reductionMember, paramSeq) \
    \
    constexpr Space prefix##ReductionSize = (reductionSize); \
    Space prefix##ReductionMember = (reductionMember); \
    \
    constexpr Space prefix##FlatSize = (outerSize) * prefix##ReductionSize; \
    Space prefix##ReductionOrigin = (outerMember) * prefix##ReductionSize; \
    Space prefix##FlatMember = prefix##ReductionOrigin + prefix##ReductionMember; \
    \
    REDUCTION_SETUPFLAT(prefix, prefix##FlatSize, prefix##FlatMember, paramSeq); \

#define REDUCTION_SETUPFLAT(prefix, flatSize, flatMember, paramSeq) \
    PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_SETUPFLAT_ITER, (prefix, paramSeq, flatSize, flatMember))

#define REDUCTION_SETUPFLAT_ITER(i, args) \
    REDUCTION_SETUPFLAT_ITER2(i, PREP_ARG4_0 args, PREP_ARG4_1 args, PREP_ARG4_2 args, PREP_ARG4_3 args)

#define REDUCTION_SETUPFLAT_ITER2(i, prefix, paramSeq, flatSize, flatMember) \
    REDUCTION_SETUPFLAT_ITER3(PREP_SEQ_ELEM(i, paramSeq), prefix, flatSize, flatMember)

#define REDUCTION_SETUPFLAT_ITER3(param, prefix, flatSize, flatMember) \
    REDUCTION_SETUPFLAT_ITER4(PREP_ARG2_0 param, PREP_ARG2_1 param, prefix, flatSize, flatMember)

#define REDUCTION_SETUPFLAT_ITER4(Type, name, prefix, flatSize, flatMember) \
    \
    devSramArray(PREP_PASTE3(prefix, name, ReductionArray), Type, flatSize); \
    \
    auto PREP_PASTE3(prefix, name, ReductionPtr) = PREP_PASTE3(prefix, name, ReductionArray) + (flatMember);

//================================================================
//
// REDUCTION_STORE
//
//================================================================

#define REDUCTION_STORE(prefix, paramSeq) \
    \
    PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_STOREFLAT_ITER, (prefix, paramSeq)) \
    \
    devSyncThreads(); \

#define REDUCTION_STOREFLAT_ITER(i, args) \
    REDUCTION_STOREFLAT_ITER2(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define REDUCTION_STOREFLAT_ITER2(i, prefix, paramSeq) \
    REDUCTION_STOREFLAT_ITER3(PREP_SEQ_ELEM(i, paramSeq), prefix)

#define REDUCTION_STOREFLAT_ITER3(param, prefix) \
    REDUCTION_STOREFLAT_ITER4(PREP_ARG2_0 param, PREP_ARG2_1 param, prefix)

#define REDUCTION_STOREFLAT_ITER4(Type, name, prefix) \
    *PREP_PASTE3(prefix, name, ReductionPtr) = name;

//================================================================
//
// REDUCTION_READBACK
//
//================================================================

#define REDUCTION_READBACK(prefix, paramSeq) \
    PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_READBACK_ITER, (prefix, paramSeq))

#define REDUCTION_READBACK_ITER(i, args) \
    REDUCTION_READBACK_ITER2(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define REDUCTION_READBACK_ITER2(i, prefix, paramSeq) \
    REDUCTION_READBACK_ITER3(PREP_SEQ_ELEM(i, paramSeq), prefix)

#define REDUCTION_READBACK_ITER3(param, prefix) \
    REDUCTION_READBACK_ITER4(PREP_ARG2_0 param, PREP_ARG2_1 param, prefix)

#define REDUCTION_READBACK_ITER4(Type, name, prefix) \
    name = PREP_PASTE3(prefix, name, ReductionArray)[PREP_PASTE(prefix, ReductionOrigin)];

//================================================================
//
// REDUCTION_ACCUMULATE
//
//================================================================

#define REDUCTION_ACCUMULATE(prefix, paramSeq, accumBody) \
    PARALLEL_REDUCTION_TWICE(prefix##ReductionSize, prefix##ReductionMember, REDUCTION_ACCUM_ITER, (prefix, paramSeq, accumBody))

#define REDUCTION_ACCUM_ITER(activeFlag, stageSize, args) \
    REDUCTION_ACCUM_ITER_EX(activeFlag, stageSize, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define REDUCTION_ACCUM_ITER_EX(activeFlag, stageSize, prefix, paramSeq, accumBody) \
    { \
        bool active = (activeFlag); \
        PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_ACCUM_VARS, (prefix, paramSeq, activeFlag, stageSize)) \
        {accumBody;} \
        devSyncThreads(); \
    }

//----------------------------------------------------------------

#define REDUCTION_ACCUM_VARS(i, args) \
    REDUCTION_ACCUM_VARS1(i, PREP_ARG4_0 args, PREP_ARG4_1 args, PREP_ARG4_2 args, PREP_ARG4_3 args)

#define REDUCTION_ACCUM_VARS1(i, prefix, paramSeq, activeFlag, stageSize) \
    REDUCTION_ACCUM_VARS2(PREP_SEQ_ELEM(i, paramSeq), prefix, activeFlag, stageSize)

#define REDUCTION_ACCUM_VARS2(typeName, prefix, activeFlag, stageSize) \
    REDUCTION_ACCUM_VARS3(PREP_ARG2_0 typeName, PREP_ARG2_1 typeName, prefix, activeFlag, stageSize)

#define REDUCTION_ACCUM_VARS3(Type, name, prefix, activeFlag, stageSize) \
    auto PREP_PASTE(name, L) = PREP_PASTE3(prefix, name, ReductionPtr); \
    auto PREP_PASTE(name, R) = PREP_PASTE3(prefix, name, ReductionPtr) + (stageSize); /* make it const to avoid user mistakes */

//================================================================
//
// REDUCTION_MAKE
//
//================================================================

#define REDUCTION_APPLY(prefix, paramSeq, accumBody) \
    \
    REDUCTION_STORE(prefix, paramSeq) \
    \
    REDUCTION_ACCUMULATE(prefix, paramSeq, accumBody) \
    \
    REDUCTION_READBACK(prefix, paramSeq)

#define REDUCTION_MAKE(prefix, outerSize, outerMember, reductionSize, reductionMember, paramSeq, accumBody) \
    REDUCTION_PREPARE(prefix, outerSize, outerMember, reductionSize, reductionMember, paramSeq) \
    REDUCTION_APPLY(prefix, paramSeq, accumBody)
