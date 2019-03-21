#pragma once

//================================================================
//
// PARALLEL_REDUCTION2
//
//================================================================

#define PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, stageSize) \
    { \
        enum {_activeCount = COMPILE_CLAMP((reductSize) - stageSize, 0, stageSize)}; \
        if (_activeCount) {bool active = (reductMember) < _activeCount; actionMacro(active, stageSize, actionContext)}; \
        devSyncThreads(); \
    }

#define PARALLEL_REDUCTION2(reductSize, reductMember, actionMacro, actionContext) \
    \
    devSyncThreads(); \
    \
    COMPILE_ASSERT((reductSize) <= 256); \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 128) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 64) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 32) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 16) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 8) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 4) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 2) \
    PARALLEL_REDUCTION2_ITER(reductSize, reductMember, actionMacro, actionContext, 1)

//================================================================
//
// REDUCTION_SETUP
//
//================================================================

#define REDUCTION_SETUP(prefix, outerSize, outerMember, reductSize, reductMember, paramSeq) \
    \
    const Space prefix##ReductSize = (reductSize); \
    Space prefix##ReductMember = (reductMember); \
    \
    const Space prefix##FlatSize = (outerSize) * prefix##ReductSize; \
    Space prefix##ReductStart = (outerMember) * (reductSize); \
    Space prefix##FlatMember = prefix##ReductStart + prefix##ReductMember; \
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
    devSramArray(PREP_PASTE3(prefix, name, ReductArray), Type, flatSize); \
    ArrayPtr(Type) PREP_PASTE3(prefix, name, ReductPtr) = &PREP_PASTE3(prefix, name, ReductArray)[flatMember];

//================================================================
//
// REDUCTION_STORE
//
//================================================================

#define REDUCTION_STORE(prefix, paramSeq) \
    PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_STOREFLAT_ITER, (prefix, paramSeq))

#define REDUCTION_STOREFLAT_ITER(i, args) \
    REDUCTION_STOREFLAT_ITER2(i, PREP_ARG2_0 args, PREP_ARG2_1 args)

#define REDUCTION_STOREFLAT_ITER2(i, prefix, paramSeq) \
    REDUCTION_STOREFLAT_ITER3(PREP_SEQ_ELEM(i, paramSeq), prefix)

#define REDUCTION_STOREFLAT_ITER3(param, prefix) \
    REDUCTION_STOREFLAT_ITER4(PREP_ARG2_0 param, PREP_ARG2_1 param, prefix)

#define REDUCTION_STOREFLAT_ITER4(Type, name, prefix) \
    *PREP_PASTE3(prefix, name, ReductPtr) = name;

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
    name = PREP_PASTE3(prefix, name, ReductArray)[PREP_PASTE(prefix, ReductStart)];

//================================================================
//
// REDUCTION_REDUCTION
//
//================================================================

#define REDUCTION_ACCUM(prefix, paramSeq, accumBody) \
    PARALLEL_REDUCTION2(prefix##ReductSize, prefix##ReductMember, REDUCTION_ACCUM_ITER, (prefix, paramSeq, accumBody))

#define REDUCTION_ACCUM_ITER(active, ofs, args) \
    REDUCTION_ACCUM_ITER_EX(active, ofs, PREP_ARG3_0 args, PREP_ARG3_1 args, PREP_ARG3_2 args)

#define REDUCTION_ACCUM_ITER_EX(active, ofs, prefix, paramSeq, accumBody) \
    { \
        PREP_FOR(PREP_SEQ_SIZE(paramSeq), REDUCTION_ACCUM_PREPARE, (prefix, paramSeq, active, ofs)) \
        {accumBody;} \
    }

#define REDUCTION_ACCUM_PREPARE(i, args) \
    REDUCTION_ACCUM_PREPARE1(i, PREP_ARG4_0 args, PREP_ARG4_1 args, PREP_ARG4_2 args, PREP_ARG4_3 args)

#define REDUCTION_ACCUM_PREPARE1(i, prefix, paramSeq, active, ofs) \
    REDUCTION_ACCUM_PREPARE2(PREP_SEQ_ELEM(i, paramSeq), prefix, active, ofs)

#define REDUCTION_ACCUM_PREPARE2(typeName, prefix, active, ofs) \
    REDUCTION_ACCUM_PREPARE3(PREP_ARG2_0 typeName, PREP_ARG2_1 typeName, prefix, active, ofs)

#define REDUCTION_ACCUM_PREPARE3(Type, name, prefix, active, ofs) \
    ArrayPtr(Type) PREP_PASTE(name, L) = PREP_PASTE3(prefix, name, ReductPtr); \
    ArrayPtr(const Type) PREP_PASTE(name, R) = PREP_PASTE3(prefix, name, ReductPtr) + ofs;

//================================================================
//
// REDUCTION_SIMPLE
//
//================================================================

#define REDUCTION_SIMPLE_MAIN(prefix, paramSeq, accumBody) \
    REDUCTION_STORE(prefix, paramSeq) \
    REDUCTION_ACCUM(prefix, paramSeq, accumBody) \
    REDUCTION_READBACK(prefix, paramSeq)

#define REDUCTION_SIMPLE(prefix, outerSize, outerMember, reductSize, reductMember, paramSeq, accumBody) \
    REDUCTION_SETUP(prefix, outerSize, outerMember, reductSize, reductMember, paramSeq) \
    REDUCTION_SIMPLE_MAIN(prefix, paramSeq, accumBody)
