#pragma once

#include "gpuSupport/parallelReduction.h"
#include "prepTools/prepSeqForeach.h"

//================================================================
//
// REDUCECLASSIC__PREPARE
//
// Allocates SRAM arrays for all enumerated parameters,
// each array for N=outerSize simultaneous reductions.
//
// Flat size and flat member are total ones for the flattened arrays.
//
// Each setup has a pre and may be used multiple times.
//
//================================================================

#define REDUCECLASSIC__PREPARE(pre, outerSize, outerMember, reductionSize, reductionMember, paramSeq) \
    \
    constexpr Space pre##ReductionSize = (reductionSize); \
    const Space pre##ReductionMember = (reductionMember); \
    \
    constexpr Space pre##FlatSize = (outerSize) * pre##ReductionSize; \
    const Space pre##ReductionOrigin = (outerMember) * pre##ReductionSize; \
    const Space pre##FlatMember = pre##ReductionOrigin + pre##ReductionMember; \
    \
    REDUCECLASSIC__SETUPFLAT(paramSeq, pre); \

//----------------------------------------------------------------

#define REDUCECLASSIC__SETUPFLAT(paramSeq, pre) \
    PREP_SEQ_FOREACH_PAIR(paramSeq, REDUCECLASSIC__SETUPFLAT_ITER, pre)

#define REDUCECLASSIC__SETUPFLAT_ITER(Type, name, pre) \
    devSramArray(pre##name##ReductionArray, Type, pre##FlatSize); \
    const auto pre##name##ReductionPtr = pre##name##ReductionArray + pre##FlatMember;

//================================================================
//
// REDUCECLASSIC__STORE
//
//================================================================

#define REDUCECLASSIC__STORE(pre, paramSeq) \
    { \
        PREP_SEQ_FOREACH_PAIR(paramSeq, REDUCECLASSIC__STOREFLAT_ITER, pre) \
        devSyncThreads(); \
    }

#define REDUCECLASSIC__STOREFLAT_ITER(Type, name, pre) \
    *pre##name##ReductionPtr = name;

//================================================================
//
// REDUCECLASSIC__READBACK
//
//================================================================

#define REDUCECLASSIC__READBACK(pre, paramSeq) \
    { \
        PREP_SEQ_FOREACH_PAIR(paramSeq, REDUCECLASSIC__READBACK_ITER, pre) \
    }

#define REDUCECLASSIC__READBACK_ITER(Type, name, pre) \
    name = pre##name##ReductionArray[pre##ReductionOrigin];

//================================================================
//
// REDUCECLASSIC__ACCUMULATE
//
//================================================================

#define REDUCECLASSIC__ACCUMULATE(pre, paramSeq, accumBody) \
    { \
        PARALLEL_REDUCTION_TWICE(pre##ReductionSize, pre##ReductionMember, REDUCECLASSIC__ACCUM_ITER, (pre, paramSeq, accumBody)) \
    }

#define REDUCECLASSIC__ACCUM_ITER(activeFlag, stageSize, args) \
    REDUCECLASSIC__ACCUM_ITER_EX(activeFlag, stageSize, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define REDUCECLASSIC__ACCUM_ITER_EX(activeFlag, stageSize, pre, paramSeq, accumBody) \
    { \
        const bool active = (activeFlag); \
        PREP_SEQ_FOREACH_PAIR(paramSeq, REDUCECLASSIC__ACCUM_VARS, (pre, stageSize)) \
        {accumBody;} \
        devSyncThreads(); \
    }

//----------------------------------------------------------------

#define REDUCECLASSIC__ACCUM_VARS(Type, name, args) \
    REDUCECLASSIC__ACCUM_VARS1(Type, name, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define REDUCECLASSIC__ACCUM_VARS1(Type, name, pre, stageSize) \
    REDUCECLASSIC__ACCUM_VARS2(Type, name, pre, stageSize)

#define REDUCECLASSIC__ACCUM_VARS2(Type, name, pre, stageSize) \
    const auto name##L = pre##name##ReductionPtr; \
    const auto name##R = pre##name##ReductionPtr + (stageSize);

//================================================================
//
// API
//
//================================================================

#define REDUCTION_CLASSIC_PREPARE_EX(pre, outerSize, outerMember, reductionSize, reductionMember, paramSeq) \
    REDUCECLASSIC__PREPARE(pre, outerSize, outerMember, reductionSize, reductionMember, paramSeq)

#define REDUCTION_CLASSIC_PREPARE(pre, reductionSize, reductionMember, paramSeq) \
    REDUCTION_CLASSIC_PREPARE_EX(pre, 1, 0, reductionSize, reductionMember, paramSeq)

//----------------------------------------------------------------

#define REDUCTION_CLASSIC_APPLY(pre, paramSeq, accumBody) \
    { \
        REDUCECLASSIC__STORE(pre, paramSeq) \
        REDUCECLASSIC__ACCUMULATE(pre, paramSeq, accumBody) \
        REDUCECLASSIC__READBACK(pre, paramSeq) \
    }

//----------------------------------------------------------------

#define REDUCTION_CLASSIC_MAKE_EX(pre, outerSize, outerMember, reductionSize, reductionMember, paramSeq, accumBody) \
    { \
        REDUCTION_CLASSIC_PREPARE_EX(pre, outerSize, outerMember, reductionSize, reductionMember, paramSeq) \
        REDUCTION_CLASSIC_APPLY(pre, paramSeq, accumBody) \
    }
