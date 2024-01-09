#pragma once

#include "numbers/interface/numberInterface.h"

//================================================================
//
// UniformPartition
//
//================================================================

template <typename Type>
class UniformPartition
{

public:

    sysinline UniformPartition()
    {
    }

    sysinline UniformPartition(const Type& size, const Type& count)
    {
        baseSize = makeUnsigned(size) / makeUnsigned(count);
        baseRem = size - baseSize * count;
    }

    sysinline Type nthOrg(const Type& index) const
    {
        return index * baseSize + clampMax(index, baseRem);
    }

    sysinline Type nthSize(const Type& index) const
    {
        return baseSize + convertExact<Type>(index < baseRem);
    }

private:

    Type baseSize = 0;
    Type baseRem = 0;

};
