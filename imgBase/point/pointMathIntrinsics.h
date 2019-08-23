#pragma once

#include "point/point.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// ldexp
//
//================================================================

template <typename Type>
sysinline Point<Type> ldexp(const Point<Type>& A, const Point<int>& B)
{
    using namespace std;
    return point(ldexp(A.X, B.X), ldexp(A.Y, B.Y));
}

template <typename Type>
sysinline Point<Type> ldexp(const Type& A, const Point<int>& B)
{
    using namespace std;
    return point(ldexp(A, B.X), ldexp(A, B.Y));
}

template <typename Type>
sysinline Point<Type> ldexp(const Point<Type>& A, const int& B)
{
    using namespace std;
    return point(ldexp(A.X, B), ldexp(A.Y, B));
}

//================================================================
//
// linerp2D
//
//================================================================

template <typename Selector, typename Value>
sysinline Value linerp2D(const Point<Selector>& condition, const Value& v00, const Value& v01, const Value& v10, const Value& v11)
{
    return linerp
    (
        condition.Y,
        linerp(condition.X, v00, v10),
        linerp(condition.X, v01, v11)
    );
}

//================================================================
//
// Funcs
//
//================================================================

POINT_DEFINE_FUNC1(saturate)

POINT_DEFINE_FUNC3(linerp)
POINT_DEFINE_FUNC3(linearIf)

//================================================================
//
// Division
//
//================================================================

POINT_DEFINE_FUNC1(nativeRecip)
POINT_DEFINE_FUNC1(nativeRecipZero)
POINT_DEFINE_FUNC2(nativeDivide)

//================================================================
//
// sqrt
//
//================================================================

POINT_DEFINE_FUNC1(fastSqrt)
POINT_DEFINE_FUNC1(recipSqrt)
POINT_DEFINE_FUNC1(sqrtf)

//================================================================
//
// pow2/log2
//
//================================================================

POINT_DEFINE_FUNC1(nativePow2)
POINT_DEFINE_FUNC1(nativeLog2)
