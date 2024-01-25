#pragma once

#include "point3d/point3dBase.h"
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

VECTOR_BASE_REBASE_VECTOR_IMPL(Point3D)
TYPE_CONTROL_VECTOR_IMPL(Point3D)

//================================================================
//
// VectorExtendImpl<Point3D>
//
//================================================================

template <typename Type>
struct VectorExtendImpl<Point3D<Type>>
{
    static sysinline Point3D<Type> func(const Type& value)
    {
        return point3D(value, value, value);
    }
};

//================================================================
//
// def<Point3D>
//
//================================================================

template <typename Type>
struct DefImpl<Point3D<Type>>
{
    static sysinline Point3D<bool> func(const Point3D<Type>& value)
    {
        return Point3D<bool>
        {
            def(value.X),
            def(value.Y),
            def(value.Z)
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
sysinline auto operator +(const Point3D<Type>& P)
    {return P;}

template <typename Type>
sysinline auto operator -(const Point3D<Type>& P)
    {return point3D(-P.X, -P.Y, -P.Z);}

template <typename Type>
sysinline auto operator !(const Point3D<Type>& P)
    {return point3D(!P.X, !P.Y, !P.Z);}

//================================================================
//
// Binary operators.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point3D<TypeA>& A, const Point3D<TypeB>& B) \
        {return point3D(A.X OP B.X, A.Y OP B.Y, A.Z OP B.Z);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point3D<TypeA>& A, const TypeB& B) \
        {return point3D(A.X OP B, A.Y OP B, A.Z OP B);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const TypeA& A, const Point3D<TypeB>& B) \
        {return point3D(A OP B.X, A OP B.Y, A OP B.Z);}

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
    sysinline auto& operator OP(Point3D<TypeA>& A, const Point3D<TypeB>& B) \
    { \
        A.X OP B.X; \
        A.Y OP B.Y; \
        A.Z OP B.Z; \
        return A; \
    } \
    template <typename TypeA, typename TypeB> \
    sysinline auto& operator OP(Point3D<TypeA>& A, const TypeB& B) \
    { \
        A.X OP B; \
        A.Y OP B; \
        A.Z OP B; \
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
sysinline bool allv(const Point3D<Type>& P)
    {return allv(P.X) && allv(P.Y) && allv(P.Z);}

template <typename Type>
sysinline bool anyv(const Point3D<Type>& P)
    {return anyv(P.X) || anyv(P.Y) || anyv(P.Z);}

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
// ConvertFamilyImpl<Point3D<T>>
//
//================================================================

CONVERT_FAMILY_VECTOR_IMPL(Point3D, Point3DFamily)

//================================================================
//
// ConvertImpl
//
// Point3D -> Point3D
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<Point3DFamily, Point3DFamily, check, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Point3D<DstBase> func(const Point3D<SrcBase>& srcPoint)
        {
            return point3D
            (
                BaseImpl::func(srcPoint.X),
                BaseImpl::func(srcPoint.Y),
                BaseImpl::func(srcPoint.Z)
            );
        }
    };
};

//================================================================
//
// ConvertImplFlag<Point3D, Point3D>
//
//================================================================

template <Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<Point3DFamily, Point3DFamily, rounding, hint>
{
    template <typename SrcPoint, typename DstPoint>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcPoint);
        using DstBase = VECTOR_BASE(DstPoint);

        using BaseImpl = typename ConvertScalarFlag<SrcBase, DstBase, rounding, hint>::Code;

        static sysinline Point3D<bool> func(const Point3D<SrcBase>& src, Point3D<DstBase>& dst)
        {
            return point3D
            (
                BaseImpl::func(src.X, dst.X),
                BaseImpl::func(src.Y, dst.Y),
                BaseImpl::func(src.Z, dst.Z)
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
sysinline void exchange(Point3D<Type>& A, Point3D<Type>& B)
{
    exchange(A.X, B.X);
    exchange(A.Y, B.Y);
    exchange(A.Z, B.Z);
}

//================================================================
//
// POINT3D_DEFINE_FUNC1
// POINT3D_DEFINE_FUNC2
// POINT3D_DEFINE_FUNC3
//
//================================================================

#define POINT3D_DEFINE_FUNC1(func) \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& P) \
        {return point3D(func(P.X), func(P.Y), func(P.Z));}

//----------------------------------------------------------------

#define POINT3D_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Point3D<Type>& B) \
        {return point3D(func(A.X, B.X), func(A.Y, B.Y), func(A.Z, B.Z));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point3D<Type>& B) \
        {return point3D(func(A, B.X), func(A, B.Y), func(A, B.Z));} \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Type& B) \
        {return point3D(func(A.X, B), func(A.Y, B), func(A.Z, B));}

//----------------------------------------------------------------

#define POINT3D_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Point3D<Type>& B, const Point3D<Type>& C) \
        {return point3D(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y), func(A.Z, B.Z, C.Z));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point3D<Type>& B, const Point3D<Type>& C) \
        {return point3D(func(A, B.X, C.X), func(A, B.Y, C.Y), func(A, B.Z, C.Z));} \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Type& B, const Point3D<Type>& C) \
        {return point3D(func(A.X, B, C.X), func(A.Y, B, C.Y), func(A.Z, B, C.Z));} \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Point3D<Type>& B, const Type& C) \
        {return point3D(func(A.X, B.X, C), func(A.Y, B.Y, C), func(A.Z, B.Z, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Point3D<Type>& A, const Type& B, const Type& C) \
        {return point3D(func(A.X, B, C), func(A.Y, B, C), func(A.Z, B, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point3D<Type>& B, const Type& C) \
        {return point3D(func(A, B.X, C), func(A, B.Y, C), func(A, B.Z, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Type& B, const Point3D<Type>& C) \
        {return point3D(func(A, B, C.X), func(A, B, C.Y), func(A, B, C.Z));}

//================================================================
//
// minv
// maxv
// clampMin
// clampMax
// clampRange
//
//================================================================

POINT3D_DEFINE_FUNC2(minv)
POINT3D_DEFINE_FUNC2(maxv)
POINT3D_DEFINE_FUNC2(clampMin)
POINT3D_DEFINE_FUNC2(clampMax)
POINT3D_DEFINE_FUNC3(clampRange)

//================================================================
//
// absv
// floor
// ceil
//
//================================================================

POINT3D_DEFINE_FUNC1(absv)
POINT3D_DEFINE_FUNC1(floorv)
POINT3D_DEFINE_FUNC1(ceilv)

//================================================================
//
// scalarProd
//
//================================================================

template <typename Float>
sysinline Float scalarProd(const Point3D<Float>& A, const Point3D<Float>& B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
}

//================================================================
//
// vectorDecompose
//
//================================================================

template <typename Float>
sysinline void vectorDecompose(const Point3D<Float>& vec, Float& vectorLengthSq, Float& vectorDivLen, Float& vectorLength, Point3D<Float>& vectorDir)
{
    vectorLengthSq = square(vec.X) + square(vec.Y) + square(vec.Z);
    vectorDivLen = recipSqrt(vectorLengthSq);
    vectorLength = vectorLengthSq * vectorDivLen;
    vectorDir = vec * vectorDivLen;

    if (vectorLengthSq == 0)
    {
        vectorLength = 0;

        vectorDir.X = 1;
        vectorDir.Y = 0;
        vectorDir.Z = 0;
    }
}

//================================================================
//
// vectorSum
//
//================================================================

template <typename Float>
sysinline Float vectorSum(const Point3D<Float>& vec)
{
    return vec.X + vec.Y + vec.Z;
}
