#pragma once

#include "point4d/point4dBase.h"
#include "numbers/interface/numberInterface.h"

//================================================================
//
// Point4D<T>
//
// Usage: the same as Point<T>
//
//================================================================

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
// VectorBaseImpl<Point4D>
//
//================================================================

template <typename Type>
struct VectorBaseImpl< Point4D<Type> >
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<Point4D>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl< Point4D<OldBase>, NewBase >
{
    using T = Point4D<NewBase>;
};

//================================================================
//
// VectorExtendImpl<Point4D>
//
//================================================================

template <typename Type>
struct VectorExtendImpl< Point4D<Type> >
{
    static sysinline Point4D<Type> func(const Type& value)
    {
        return point4D(value, value, value, value);
    }
};

//================================================================
//
// def<Point4D>
//
//================================================================

template <typename Type>
struct DefImpl<Point4D<Type>>
{
    static sysinline Point4D<bool> func(const Point4D<Type>& value)
    {
        return Point4D<bool>{def(value.X), def(value.Y), def(value.Z), def(value.W)};
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
sysinline Point4D<Type> operator +(const Point4D<Type>& P)
    {return P;}

template <typename Type>
sysinline Point4D<Type> operator -(const Point4D<Type>& P)
    {return point4D(Type(-P.X), Type(-P.Y), Type(-P.Z), Type(-P.W));}

//================================================================
//
// Arithmetic binary operations: +, -, *, etc.
//
//================================================================

#define TMP_MACRO(op) \
    \
    template <typename Type> \
    sysinline Point4D<Type> operator op(const Point4D<Type>& A, const Point4D<Type>& B) \
        {return point4D(A.X op B.X, A.Y op B.Y, A.Z op B.Z, A.W op B.W);} \
    \
    template <typename Type> \
    sysinline Point4D<Type> operator op(const Point4D<Type>& A, const Type& B) \
        {return point4D(A.X op B, A.Y op B, A.Z op B, A.W op B);} \
    \
    template <typename Type> \
    sysinline Point4D<Type> operator op(const Type& A, const Point4D<Type>& B) \
        {return point4D(A op B.X, A op B.Y, A op B.Z, A op B.W);}

TMP_MACRO(+)
TMP_MACRO(-)
TMP_MACRO(*)
TMP_MACRO(/)
TMP_MACRO(%)
TMP_MACRO(>>)
TMP_MACRO(<<)
TMP_MACRO(&)
TMP_MACRO(|)

#undef TMP_MACRO

//================================================================
//
// Assignment operations on Point4D type: +=, -=, etc.
//
//================================================================

#define TMP_MACRO(Result, asgop) \
    \
    template <typename Type> \
    sysinline Point4D<Result>& operator asgop(Point4D<Type>& A, const Point4D<Type>& B) \
    { \
        A.X asgop B.X; \
        A.Y asgop B.Y; \
        A.Z asgop B.Z; \
        A.W asgop B.W; \
        return A; \
    } \
    template <typename Type> \
    sysinline Point4D<Result>& operator asgop(Point4D<Type>& A, const Type& B) \
    { \
        A.X asgop B; \
        A.Y asgop B; \
        A.Z asgop B; \
        A.W asgop B; \
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
// Vector bool comparisons for Point4D<T>: ==, !=, <, >, <=, >=
// The result is Point4D<bool>.
//
//================================================================

#define TMP_MACRO(Result, op) \
    \
    template <typename Type> \
    sysinline Point4D<Result> operator op(const Point4D<Type>& A, const Point4D<Type>& B) \
        {return point4D(A.X op B.X, A.Y op B.Y, A.Z op B.Z, A.W op B.W);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point4D<Result> operator op(const Point4D<Type>& A, const Scalar& B) \
        {return point4D(A.X op B, A.Y op B, A.Z op B, A.W op B);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point4D<Result> operator op(const Scalar& A, const Point4D<Type>& B) \
        {return point4D(A op B.X, A op B.Y, A op B.Z, A op B.W);}

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
// Input and output is Point4D<bool>.
//
//================================================================

sysinline Point4D<bool> operator !(const Point4D<bool>& P)
    {return point4D(!P.X, !P.Y, !P.Z, !P.W);}

#define TMP_MACRO(op) \
    \
    sysinline Point4D<bool> operator op(const Point4D<bool>& A, const Point4D<bool>& B) \
        {return point4D(A.X op B.X, A.Y op B.Y, A.Z op B.Z, A.W op B.W);} \
    \
    sysinline Point4D<bool> operator op(const Point4D<bool>& A, bool B) \
        {return point4D(A.X op B, A.Y op B, A.Z op B, A.W op B);} \
    \
    sysinline Point4D<bool> operator op(bool A, const Point4D<bool>& B) \
        {return point4D(A op B.X, A op B.Y, A op B.Z, A op B.W);}

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

sysinline bool allv(const Point4D<bool>& P)
    {return P.X && P.Y && P.Z && P.W;}

sysinline bool anyv(const Point4D<bool>& P)
    {return P.X || P.Y || P.Z || P.W;}

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
// ConvertFamilyImpl< Point4D<T> >
//
//================================================================

struct Point4DFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl< Point4D<Type> >
{
    using T = Point4DFamily;
};

//================================================================
//
// ConvertImpl
//
// Point4D -> Point4D
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<Point4DFamily, Point4DFamily, check, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Point4D<DstBase> func(const Point4D<SrcBase>& srcPoint)
        {
            return point4D(BaseImpl::func(srcPoint.X), BaseImpl::func(srcPoint.Y), BaseImpl::func(srcPoint.Z), BaseImpl::func(srcPoint.W));
        }
    };
};

//================================================================
//
// ConvertImplFlag<Point4D, Point4D>
//
//================================================================

template <Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<Point4DFamily, Point4DFamily, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalarFlag<SrcBase, DstBase, rounding, hint>::Code;

        static sysinline Point4D<bool> func(const Point4D<SrcBase>& src, Point4D<DstBase>& dst)
        {
            bool sX = BaseImpl::func(src.X, dst.X);
            bool sY = BaseImpl::func(src.Y, dst.Y);
            bool sZ = BaseImpl::func(src.Z, dst.Z);
            bool sW = BaseImpl::func(src.W, dst.W);

            return point4D(sX, sY, sZ, sW);
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
// POINT4D_DEFINE_FUNC1
// POINT4D_DEFINE_FUNC2
// POINT4D_DEFINE_FUNC3
//
//================================================================

#define POINT4D_DEFINE_FUNC1(func) \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& P) \
        {return point4D(func(P.X), func(P.Y), func(P.Z), func(P.W));} \

//----------------------------------------------------------------

#define POINT4D_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Point4D<Type>& B) \
        {return point4D(func(A.X, B.X), func(A.Y, B.Y), func(A.Z, B.Z), func(A.W, B.W));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Type& A, const Point4D<Type>& B) \
        {return point4D(func(A, B.X), func(A, B.Y), func(A, B.Z), func(A, B.W));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Type& B) \
        {return point4D(func(A.X, B), func(A.Y, B), func(A.Z, B), func(A.W, B));}

//----------------------------------------------------------------

#define POINT4D_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Point4D<Type>& B, const Point4D<Type>& C) \
        {return point4D(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y), func(A.Z, B.Z, C.Z), func(A.W, B.W, C.W));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Type& A, const Point4D<Type>& B, const Point4D<Type>& C) \
        {return point4D(func(A, B.X, C.X), func(A, B.Y, C.Y), func(A, B.Z, C.Z), func(A, B.W, C.W));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Type& B, const Point4D<Type>& C) \
        {return point4D(func(A.X, B, C.X), func(A.Y, B, C.Y), func(A.Z, B, C.Z), func(A.W, B, C.W));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Point4D<Type>& B, const Type& C) \
        {return point4D(func(A.X, B.X, C), func(A.Y, B.Y, C), func(A.Z, B.Z, C), func(A.W, B.W, C));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Point4D<Type>& A, const Type& B, const Type& C) \
        {return point4D(func(A.X, B, C), func(A.Y, B, C), func(A.Z, B, C), func(A.W, B, C));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Type& A, const Point4D<Type>& B, const Type& C) \
        {return point4D(func(A, B.X, C), func(A, B.Y, C), func(A, B.Z, C), func(A, B.W, C));} \
    \
    template <typename Type> \
    sysinline Point4D<Type> func(const Type& A, const Type& B, const Point4D<Type>& C) \
        {return point4D(func(A, B, C.X), func(A, B, C.Y), func(A, B, C.Z), func(A, B, C.W));} \

//================================================================
//
// minv
// maxv
// clampMin
// clampMax
// clampRange
//
//================================================================

POINT4D_DEFINE_FUNC2(minv)
POINT4D_DEFINE_FUNC2(maxv)
POINT4D_DEFINE_FUNC2(clampMin)
POINT4D_DEFINE_FUNC2(clampMax)
POINT4D_DEFINE_FUNC3(clampRange)

//================================================================
//
// absv
// sqrtf
//
//================================================================

POINT4D_DEFINE_FUNC1(absv)
POINT4D_DEFINE_FUNC1(sqrtf)
