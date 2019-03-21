#pragma once

#include "numbers/interface/numberInterface.h"
#include "numbers/divRoundCompile.h"

//================================================================
//
// divDownNonneg
// divUpNonneg
// divNearestNonneg
//
// (no overflow checking)
//
//================================================================

template <typename Type>
sysinline Type divDownNonneg(const Type& A, const Type& B)
    {return A / B;}

template <typename Type>
sysinline Type divUpNonneg(const Type& A, const Type& B)
    {return (A + (B - convertExact<Type>(1))) / B;}

template <typename Type>
sysinline Type divNearestNonneg(const Type& A, const Type& B)
    {return (A + (B >> 1)) / B;}

//================================================================
//
// divDown
// divUp
//
//================================================================

template <typename Type>
sysinline Type divDown(const Type& A, const Type& B)
    {return (A >= convertNearest<Type>(0)) ? divDownNonneg(A, B) : -divUpNonneg(-A, B);}

template <typename Type>
sysinline Type divUp(const Type& A, const Type& B)
    {return (A >= convertNearest<Type>(0)) ? divUpNonneg(A, B) : -divDownNonneg(-A, B);}

//================================================================
//
// divNonneg
//
//================================================================

template <typename Type>
inline Type divNonneg(const Type& A, const Type& B, Rounding rounding)
{
    Type result = A;

    if (rounding == RoundDown)
        result = divDownNonneg(A, B);

    if (rounding == RoundUp)
        result = divUpNonneg(A, B);

    if (rounding == RoundNearest)
        result = divNearestNonneg(A, B);

    return result;
}
