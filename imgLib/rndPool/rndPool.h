#pragma once

#include "data/space.h"
#include "numbers/int/intBase.h"

//================================================================
//
// RndRange
//
//================================================================

struct RndRange
{

public:

    uint32 base;
    Space size;

public:

    sysinline RndRange()
        : base(0), size(0) {}

    sysinline RndRange(uint32 base, Space size)
        : base(base), size(size) {}

};

//================================================================
//
// RndPool
//
//================================================================

class RndPool
{

public:

    sysinline RndRange allocRange(Space size)
    {
        auto oldBase = sequenceBase;
        sequenceBase += size;
        return RndRange(oldBase, size);
    }

public:

    sysinline RndPool(uint32& sequenceBase)
        : sequenceBase(sequenceBase) {}

private:

    uint32& sequenceBase;

};
