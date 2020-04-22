#pragma once

#include "gpuSupport/parallelLoop.h"

//================================================================
//
// parallelLoadTypeNoSync
//
//================================================================

template <Space groupSize, typename LoadMode, typename Type>
sysinline void parallelLoadTypeNoSync(Space groupMember, const Type& srcParam, Type& dstParam)
{
    using Atom = uint32;

    ////

    COMPILE_ASSERT(sizeof(Type) % sizeof(Atom) == 0);
    const Space atomCount = sizeof(Type) / sizeof(Atom);
    COMPILE_ASSERT(alignof(Type) % alignof(Atom) == 0);

    ////

    // ``` use debug pointers
    const Atom* srcPtr = (const Atom*) &srcParam;
    Atom* dstPtr = (Atom*) &dstParam;

    ////

    srcPtr += groupMember;
    dstPtr += groupMember;

    ////

    PARALLEL_LOOP_UNBASED
    (
        i,
        atomCount,
        groupMember,
        groupSize,
        dstPtr[i] = LoadMode::func(&srcPtr[i])
    );
}

//================================================================
//
// parallelLoadArrayNoSync
//
// Loads a structure from DRAM to SRAM by GPU group.
//
//================================================================

template <Space groupSize, Space arraySize, typename LoadMode, typename Type>
sysinline void parallelLoadArrayNoSync(Space groupMember, const Type* srcArray, Type* dstArray)
{
    using Atom = uint32;

    ////

    constexpr Space loadSize = sizeof(Type) * arraySize;

    COMPILE_ASSERT(loadSize % sizeof(Atom) == 0);
    constexpr Space atomCount = loadSize / sizeof(Atom);
    COMPILE_ASSERT(alignof(Type) % alignof(Atom) == 0);

    ////

    // ``` use debug pointers
    const Atom* srcPtr = (const Atom*) srcArray;
    Atom* dstPtr = (Atom*) dstArray;

    ////

    srcPtr += groupMember;
    dstPtr += groupMember;

    ////

    PARALLEL_LOOP_UNBASED
    (
        i,
        atomCount,
        groupMember,
        groupSize,
        dstPtr[i] = LoadMode::func(&srcPtr[i])
    );
}
