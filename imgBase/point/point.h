#pragma once

#include "point/pointBase.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intBase.h"

//================================================================
//
// Point<T>
//
// Simple pair of X/Y values.
//
// USAGE EXAMPLES:
//
//================================================================

#if 0

// Type traits: signed/unsigned.
COMPILE_ASSERT(TYPE_IS_SIGNED(const Point<int>));
COMPILE_ASSERT(TYPE_EQUAL(TYPE_MAKE_UNSIGNED_(Point<int>), Point<unsigned>));

// Type traits: controlled support.
COMPILE_ASSERT(TYPE_IS_CONTROLLED(const Point<float>));

// Type traits: min and max.
Point<int> minPoint = typeMin< Point<int> >();

// Default constructor: uninitialized
Point<int> uninitializedValue;

// Construct by components.
Point<int> constructExample(2, 3);

// Create point by components. The scalar version creates a point
// with both components having the same value.
Point<int> A = point(16, 32);
Point<int> B = point(2); // B = {2, 2}

// Convert point->point, scalar-guided.
Point<float> c0 = convertNearest<float>(A);

// Convert point->point, vector-guided.
Point<float> c1 = convertNearest< Point<float> >(B);

// Convert scalar->point.
Point<float> c2 = convertNearest< Point<float> >(1);

// Convert with success flag.
Point<int> intResult;
Point<bool> intSuccess = convertNearest(c2, intResult);
intSuccess = convertNearest(25, intResult);

// Make zero
Point<float32> makeZero = zeroOf<Point<float32>>();

// Generate NAN
Point<float> pointNan = nanOf<const Point<float>&>();

// Is definite?
Point<bool> pointDef = def(pointNan);

// Unary operations.
Point<int> testUnary = -A;

// Arithmetic binary operations.
Point<int> C = A + B;
Point<int> D = 2 * A;

// Assignment arithmetic operations.
A += 1;
C &= D;

// Vector comparisons.
Point<bool> Q = (A == B);
Point<bool> R = (1 <= B);

// Vector bool operations.
Point<bool> Z = Q && R;
Point<bool> W = Q || R;
Point<bool> V = !Z;

// Reduction of Point<bool> to bool.
require(allv(A >= 0));
if (anyv(A < 0)) return false;

// Min, max, clampRange functions.
Point<int> t1 = minv(A, B);
Point<int> t2 = maxv(A, B);
Point<int> t3 = clampMin(A, 0);
Point<int> t4 = clampMax(A, 10);
Point<int> t5 = clampRange(A, 0, 10);

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Traits
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VectorBaseImpl<Point>
//
//================================================================

template <typename Type>
struct VectorBaseImpl< Point<Type> >
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<Point>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl< Point<OldBase>, NewBase >
{
    using T = ::Point<NewBase>;
};

//================================================================
//
// VectorExtendImpl<Point>
//
//================================================================

template <typename Type>
struct VectorExtendImpl< Point<Type> >
{
    static sysinline Point<Type> func(const Type& value)
    {
        return point(value, value);
    }
};

//================================================================
//
// def<Point>
//
//================================================================

template <typename Type>
struct DefImpl<Point<Type>>
{
    static sysinline Point<bool> func(const Point<Type>& value)
    {
        return Point<bool>{def(value.X), def(value.Y)};
    }
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Base arithmetics
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// Unary +, -.
//
//================================================================

template <typename Type>
sysinline Point<Type> operator +(const Point<Type>& P)
    {return P;}

template <typename Type>
sysinline Point<Type> operator -(const Point<Type>& P)
    {return point(Type(-P.X), Type(-P.Y));}

//================================================================
//
// Arithmetic binary operations: +, -, *, etc.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Point<Type>& A, const Point<Type>& B) \
        {return point<Type>(A.X OP B.X, A.Y OP B.Y);} \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Point<Type>& A, const Type& B) \
        {return point<Type>(A.X OP B, A.Y OP B);} \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Type& A, const Point<Type>& B) \
        {return point<Type>(A OP B.X, A OP B.Y);}

TMP_MACRO(+)
TMP_MACRO(-)
TMP_MACRO(*)
TMP_MACRO(/)
TMP_MACRO(%)
TMP_MACRO(&)
TMP_MACRO(|)

#undef TMP_MACRO

//================================================================
//
// Shifts
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Point<Type>& A, const Point<int32>& B) \
        {return point<Type>(A.X OP B.X, A.Y OP B.Y);} \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Point<Type>& A, const int32& B) \
        {return point<Type>(A.X OP B, A.Y OP B);} \
    \
    template <typename Type> \
    sysinline Point<Type> operator OP(const Type& A, const Point<int32>& B) \
        {return point<Type>(A OP B.X, A OP B.Y);}

TMP_MACRO(>>)
TMP_MACRO(<<)

#undef TMP_MACRO

//================================================================
//
// Assignment operations on Point type: +=, -=, etc.
//
//================================================================

#define TMP_MACRO(Result, OP) \
    \
    template <typename Type> \
    sysinline Point<Result>& operator OP(Point<Type>& A, const Point<Type>& B) \
    { \
        A.X OP B.X; \
        A.Y OP B.Y; \
        return A; \
    } \
    template <typename Type> \
    sysinline Point<Result>& operator OP(Point<Type>& A, const Type& B) \
    { \
        A.X OP B; \
        A.Y OP B; \
        return A; \
    }

TMP_MACRO(Type, +=)
TMP_MACRO(Type, -=)
TMP_MACRO(Type, *=)
TMP_MACRO(Type, /=)
TMP_MACRO(Type, %=)
TMP_MACRO(Type, >>=)
TMP_MACRO(Type, <<=)
TMP_MACRO(Type, &=)
TMP_MACRO(Type, |=)

#undef TMP_MACRO

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Comparisons and bool operations.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// Vector bool comparisons for Point<T>: ==, !=, <, >, <=, >=
// The result is Point<bool>.
//
//================================================================

#define TMP_MACRO(Result, OP) \
    \
    template <typename Type> \
    sysinline Point<Result> operator OP(const Point<Type>& A, const Point<Type>& B) \
        {return point(A.X OP B.X, A.Y OP B.Y);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point<Result> operator OP(const Point<Type>& A, const Scalar& B) \
        {return point(A.X OP B, A.Y OP B);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point<Result> operator OP(const Scalar& A, const Point<Type>& B) \
        {return point(A OP B.X, A OP B.Y);}

TMP_MACRO(bool, ==)
TMP_MACRO(bool, !=)
TMP_MACRO(bool, <)
TMP_MACRO(bool, >)
TMP_MACRO(bool, <=)
TMP_MACRO(bool, >=)

#undef TMP_MACRO

//================================================================
//
// Vector bool operations: !, &&, ||.
// Input and output is Point<bool>.
//
//================================================================

sysinline Point<bool> operator !(const Point<bool>& P)
    {return point(!P.X, !P.Y);}

#define TMP_MACRO(OP) \
    \
    sysinline Point<bool> operator OP(const Point<bool>& A, const Point<bool>& B) \
        {return point(A.X OP B.X, A.Y OP B.Y);} \
    \
    sysinline Point<bool> operator OP(const Point<bool>& A, bool B) \
        {return point(A.X OP B, A.Y OP B);} \
    \
    sysinline Point<bool> operator OP(bool A, const Point<bool>& B) \
        {return point(A OP B.X, A OP B.Y);}

TMP_MACRO(&&)
TMP_MACRO(||)

#undef TMP_MACRO

//================================================================
//
// allv
// Scalar "AND" of vector bool.
//
// anyv
// Scalar "OR" of vector bool.
//
//================================================================

sysinline bool allv(const Point<bool>& P)
    {return P.X && P.Y;}

sysinline bool anyv(const Point<bool>& P)
    {return P.X || P.Y;}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConvertFamilyImpl< Point<T> >
//
//================================================================

struct PointFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl< Point<Type> >
{
    using T = PointFamily;
};

//================================================================
//
// ConvertImpl
//
// Point -> Point
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<PointFamily, PointFamily, check, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Point<DstBase> func(const Point<SrcBase>& srcPoint)
        {
            return point(BaseImpl::func(srcPoint.X), BaseImpl::func(srcPoint.Y));
        }
    };
};

//================================================================
//
// ConvertImplFlag<Point, Point>
//
//================================================================

template <Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<PointFamily, PointFamily, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalarFlag<SrcBase, DstBase, rounding, hint>::Code;

        static sysinline Point<bool> func(const Point<SrcBase>& src, Point<DstBase>& dst)
        {
            bool sX = BaseImpl::func(src.X, dst.X);
            bool sY = BaseImpl::func(src.Y, dst.Y);

            return point(sX, sY);
        };
    };
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Basic utilities
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// exchange
//
//================================================================

template <typename Type>
sysinline void exchange(Point<Type>& a, Point<Type>& b)
{
    exchange(a.X, b.X);
    exchange(a.Y, b.Y);
}

//================================================================
//
// POINT_DEFINE_FUNC1
// POINT_DEFINE_FUNC2
// POINT_DEFINE_FUNC3
//
//================================================================

#define POINT_DEFINE_FUNC1(func) \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& P) \
        {return point(func(P.X), func(P.Y));} \

//----------------------------------------------------------------

#define POINT_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Point<Type>& B) \
        {return point(func(A.X, B.X), func(A.Y, B.Y));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Type& A, const Point<Type>& B) \
        {return point(func(A, B.X), func(A, B.Y));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Type& B) \
        {return point(func(A.X, B), func(A.Y, B));} \

//----------------------------------------------------------------

#define POINT_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Point<Type>& B, const Point<Type>& C) \
        {return point(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Type& A, const Point<Type>& B, const Point<Type>& C) \
        {return point(func(A, B.X, C.X), func(A, B.Y, C.Y));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Type& B, const Point<Type>& C) \
        {return point(func(A.X, B, C.X), func(A.Y, B, C.Y));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Point<Type>& B, const Type& C) \
        {return point(func(A.X, B.X, C), func(A.Y, B.Y, C));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Point<Type>& A, const Type& B, const Type& C) \
        {return point(func(A.X, B, C), func(A.Y, B, C));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Type& A, const Point<Type>& B, const Type& C) \
        {return point(func(A, B.X, C), func(A, B.Y, C));} \
    \
    template <typename Type> \
    sysinline Point<Type> func(const Type& A, const Type& B, const Point<Type>& C) \
        {return point(func(A, B, C.X), func(A, B, C.Y));} \

//================================================================
//
// minv
// maxv
// clampMin
// clampMax
// clampRange
//
//================================================================

POINT_DEFINE_FUNC2(minv)
POINT_DEFINE_FUNC2(maxv)
POINT_DEFINE_FUNC2(clampMin)
POINT_DEFINE_FUNC2(clampMax)
POINT_DEFINE_FUNC3(clampRange)

//================================================================
//
// floor
// ceil
// absv
// sqrtf
//
//================================================================

POINT_DEFINE_FUNC1(floorf)
POINT_DEFINE_FUNC1(ceilf)
POINT_DEFINE_FUNC1(absv)
POINT_DEFINE_FUNC1(sqrtf)

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
// isPower2
//
//================================================================

template <typename Type>
sysinline Point<bool> isPower2(const Point<Type>& P)
{
    return Point<bool>(isPower2(P.X), isPower2(P.Y));
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
