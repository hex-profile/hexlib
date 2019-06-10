#pragma once

#include "point4d/point4dBase.h"
#include "numbers/interface/numberInterface.h"

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
struct VectorBaseImpl<Point4D<Type>>
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<Point4D>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl<Point4D<OldBase>, NewBase>
{
    using T = Point4D<NewBase>;
};

//================================================================
//
// VectorExtendImpl<Point4D>
//
//================================================================

template <typename Type>
struct VectorExtendImpl<Point4D<Type>>
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
        return Point4D<bool>
        {
            def(value.X), 
            def(value.Y),
            def(value.Z),
            def(value.W)
        };
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
// Unary operators.
//
//================================================================

template <typename Type>
sysinline auto operator +(const Point4D<Type>& P)
    {return P;}

template <typename Type>
sysinline auto operator -(const Point4D<Type>& P)
    {return point4D(-P.X, -P.Y, -P.Z, -P.W);}

template <typename Type>
sysinline auto operator !(const Point4D<Type>& P)
    {return point4D(!P.X, !P.Y, !P.Z, !P.W);}

//================================================================
//
// Binary operators.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point4D<TypeA>& A, const Point4D<TypeB>& B) \
        {return point4D(A.X OP B.X, A.Y OP B.Y, A.Z OP B.Z, A.W OP B.W);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point4D<TypeA>& A, const TypeB& B) \
        {return point4D(A.X OP B, A.Y OP B, A.Z OP B, A.W OP B);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const TypeA& A, const Point4D<TypeB>& B) \
        {return point4D(A OP B.X, A OP B.Y, A OP B.Z, A OP B.W);}

TMP_MACRO(+)
TMP_MACRO(-)
TMP_MACRO(*)
TMP_MACRO(/)
TMP_MACRO(%)
TMP_MACRO(&)
TMP_MACRO(|)
TMP_MACRO(>>)
TMP_MACRO(<<)

TMP_MACRO(==)
TMP_MACRO(!=)
TMP_MACRO(<)
TMP_MACRO(>)
TMP_MACRO(<=)
TMP_MACRO(>=)

TMP_MACRO(&&)
TMP_MACRO(||)

#undef TMP_MACRO

//================================================================
//
// Assignment operators.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto& operator OP(Point4D<TypeA>& A, const Point4D<TypeB>& B) \
    { \
        A.X OP B.X; \
        A.Y OP B.Y; \
        A.Z OP B.Z; \
        A.W OP B.W; \
        return A; \
    } \
    template <typename TypeA, typename TypeB> \
    sysinline auto& operator OP(Point4D<TypeA>& A, const TypeB& B) \
    { \
        A.X OP B; \
        A.Y OP B; \
        A.Z OP B; \
        A.W OP B; \
        return A; \
    }

TMP_MACRO(+=)
TMP_MACRO(-=)
TMP_MACRO(*=)
TMP_MACRO(/=)
TMP_MACRO(%=)
TMP_MACRO(&=)
TMP_MACRO(|=)

TMP_MACRO(>>=)
TMP_MACRO(<<=)

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
// allv
// Scalar "AND" of vector bool.
//
// anyv
// Scalar "OR" of vector bool.
//
//================================================================

template <typename Type>
sysinline bool allv(const Point4D<Type>& P)
    {return allv(P.X) && allv(P.Y) && allv(P.Z) && allv(P.W);}

template <typename Type>
sysinline bool anyv(const Point4D<Type>& P)
    {return anyv(P.X) || anyv(P.Y) || anyv(P.Z) || anyv(P.W);}

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
// ConvertFamilyImpl<Point4D<T>>
//
//================================================================

struct Point4DFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl<Point4D<Type>>
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
            return point4D
            (
                BaseImpl::func(srcPoint.X), 
                BaseImpl::func(srcPoint.Y),
                BaseImpl::func(srcPoint.Z),
                BaseImpl::func(srcPoint.W)
            );
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
            return point4D
            (
                BaseImpl::func(src.X, dst.X),
                BaseImpl::func(src.Y, dst.Y),
                BaseImpl::func(src.Z, dst.Z),
                BaseImpl::func(src.W, dst.W)
            );
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
sysinline void exchange(Point4D<Type>& A, Point4D<Type>& B)
{
    exchange(A.X, B.X);
    exchange(A.Y, B.Y);
    exchange(A.Z, B.Z);
    exchange(A.W, B.W);
}

//================================================================
//
// POINT4D_DEFINE_FUNC1
// POINT4D_DEFINE_FUNC2
// POINT4D_DEFINE_FUNC3
//
//================================================================

#define POINT4D_DEFINE_FUNC1(func) \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& P) \
        {return point4D(func(P.X), func(P.Y), func(P.Z), func(P.W));} 

//----------------------------------------------------------------

#define POINT4D_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Point4D<Type>& B) \
        {return point4D(func(A.X, B.X), func(A.Y, B.Y), func(A.Z, B.Z), func(A.W, B.W));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point4D<Type>& B) \
        {return point4D(func(A, B.X), func(A, B.Y), func(A, B.Z), func(A, B.W));} \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Type& B) \
        {return point4D(func(A.X, B), func(A.Y, B), func(A.Z, B), func(A.W, B));}

//----------------------------------------------------------------

#define POINT4D_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Point4D<Type>& B, const Point4D<Type>& C) \
        {return point4D(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y), func(A.Z, B.Z, C.Z), func(A.W, B.W, C.W));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point4D<Type>& B, const Point4D<Type>& C) \
        {return point4D(func(A, B.X, C.X), func(A, B.Y, C.Y), func(A, B.Z, C.Z), func(A, B.W, C.W));} \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Type& B, const Point4D<Type>& C) \
        {return point4D(func(A.X, B, C.X), func(A.Y, B, C.Y), func(A.Z, B, C.Z), func(A.W, B, C.W));} \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Point4D<Type>& B, const Type& C) \
        {return point4D(func(A.X, B.X, C), func(A.Y, B.Y, C), func(A.Z, B.Z, C), func(A.W, B.W, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Point4D<Type>& A, const Type& B, const Type& C) \
        {return point4D(func(A.X, B, C), func(A.Y, B, C), func(A.Z, B, C), func(A.W, B, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point4D<Type>& B, const Type& C) \
        {return point4D(func(A, B.X, C), func(A, B.Y, C), func(A, B.Z, C), func(A, B.W, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Type& B, const Point4D<Type>& C) \
        {return point4D(func(A, B, C.X), func(A, B, C.Y), func(A, B, C.Z), func(A, B, C.W));}

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
// floor
// ceil
// absv
//
//================================================================

POINT4D_DEFINE_FUNC1(floorf)
POINT4D_DEFINE_FUNC1(ceilf)
POINT4D_DEFINE_FUNC1(absv)

//================================================================
//
// vectorLengthSq
//
//================================================================

template <typename Float>
sysinline Float vectorLengthSq(const Point4D<Float>& vec)
    {return square(vec.X) + square(vec.Y) + square(vec.Z) + square(vec.W);}

//================================================================
//
// vectorDecompose
//
//================================================================

template <typename Float>
sysinline void vectorDecompose(const Point4D<Float>& vec, Float& vectorLengthSq, Float& vectorDivLen, Float& vectorLength, Point4D<Float>& vectorDir)
{
    vectorLengthSq = square(vec.X) + square(vec.Y) + square(vec.Z) + square(vec.W);
    vectorDivLen = recipSqrt(vectorLengthSq);
    vectorLength = vectorLengthSq * vectorDivLen;
    vectorDir = vec * vectorDivLen;

    if (vectorLengthSq == 0)
    {
        vectorLength = 0;

        vectorDir.X = 1;
        vectorDir.Y = 0;
        vectorDir.Z = 0;
        vectorDir.W = 0;
    }
}
