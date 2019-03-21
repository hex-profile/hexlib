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

    UniformPartition(const Type& size, const Type& count)
    {
        baseSize = size / count;
        baseRem = size - baseSize * count;
    }

    Type nthOrg(const Type& index) const
    {
        return index * baseSize + clampMax(index, baseRem);
    }

    Type nthSize(const Type& index)
    {
        return baseSize + convertExact<Type>(index < baseRem);
    }

private:

    Type baseSize;
    Type baseRem;

};
