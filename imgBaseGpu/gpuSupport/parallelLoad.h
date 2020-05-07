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

    auto srcPtr = ArrayPtrCreate(const Atom, (const Atom*) &srcParam, atomCount, DbgptrArrayPreconditions());
    auto dstPtr = ArrayPtrCreate(Atom, (Atom*) &dstParam, atomCount, DbgptrArrayPreconditions());

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
