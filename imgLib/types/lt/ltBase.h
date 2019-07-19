#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// LinearTransform
//
// LT == "linear transform"
//
//================================================================

template <typename Type>
struct LinearTransform
{
    Type C1;
    Type C0;

    template <typename ValueType>
    sysinline auto operator () (const ValueType& value) const
        {return value * C1 + C0;}
};

//----------------------------------------------------------------

template <typename Type>
sysinline LinearTransform<Type> linearTransform(const Type& C1, const Type& C0)
{
    LinearTransform<Type> tmp;
    tmp.C0 = C0;
    tmp.C1 = C1;
    return tmp;
}
