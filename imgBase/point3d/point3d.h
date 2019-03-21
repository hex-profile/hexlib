#pragma once

#include "point3d/point3dBase.h"
#include "numbers/interface/numberInterface.h"

//================================================================
//
// Point3D<T>
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
// VectorBaseImpl<Point3D>
//
//================================================================

template <typename Type>
struct VectorBaseImpl< Point3D<Type> >
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<Point3D>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl< Point3D<OldBase>, NewBase >
{
    using T = Point3D<NewBase>;
};

//================================================================
//
// VectorExtendImpl<Point3D>
//
//================================================================

template <typename Type>
struct VectorExtendImpl< Point3D<Type> >
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
        return Point3D<bool>{def(value.X), def(value.Y), def(value.Z)};
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
sysinline Point3D<Type> operator +(const Point3D<Type>& P)
    {return P;}

template <typename Type>
sysinline Point3D<Type> operator -(const Point3D<Type>& P)
    {return point3D(Type(-P.X), Type(-P.Y), Type(-P.Z));}

//================================================================
//
// Arithmetic binary operations: +, -, *, etc.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename Type> \
    sysinline Point3D<Type> operator OP(const Point3D<Type>& A, const Point3D<Type>& B) \
        {return point3D(A.X OP B.X, A.Y OP B.Y, A.Z OP B.Z);} \
    \
    template <typename Type> \
    sysinline Point3D<Type> operator OP(const Point3D<Type>& A, const Type& B) \
        {return point3D(A.X OP B, A.Y OP B, A.Z OP B);} \
    \
    template <typename Type> \
    sysinline Point3D<Type> operator OP(const Type& A, const Point3D<Type>& B) \
        {return point3D(A OP B.X, A OP B.Y, A OP B.Z);}

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
// Assignment operations on Point3D type: +=, -=, etc.
//
//================================================================

#define TMP_MACRO(Result, ASGOP) \
    \
    template <typename Type> \
    sysinline Point3D<Result>& operator ASGOP(Point3D<Type>& A, const Point3D<Type>& B) \
    { \
        A.X ASGOP B.X; \
        A.Y ASGOP B.Y; \
        A.Z ASGOP B.Z; \
        return A; \
    } \
    template <typename Type> \
    sysinline Point3D<Result>& operator ASGOP(Point3D<Type>& A, const Type& B) \
    { \
        A.X ASGOP B; \
        A.Y ASGOP B; \
        A.Z ASGOP B; \
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
// Vector bool comparisons for Point3D<T>: ==, !=, <, >, <=, >=
// The result is Point3D<bool>.
//
//================================================================

#define TMP_MACRO(Result, OP) \
    \
    template <typename Type> \
    sysinline Point3D<Result> operator OP(const Point3D<Type>& A, const Point3D<Type>& B) \
        {return point3D(A.X OP B.X, A.Y OP B.Y, A.Z OP B.Z);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point3D<Result> operator OP(const Point3D<Type>& A, const Scalar& B) \
        {return point3D(A.X OP B, A.Y OP B, A.Z OP B);} \
    \
    template <typename Type, typename Scalar> \
    sysinline Point3D<Result> operator OP(const Scalar& A, const Point3D<Type>& B) \
        {return point3D(A OP B.X, A OP B.Y, A OP B.Z);}

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
// Input and output is Point3D<bool>.
//
//================================================================

sysinline Point3D<bool> operator !(const Point3D<bool>& P)
    {return point3D(!P.X, !P.Y, !P.Z);}

#define TMP_MACRO(OP) \
    \
    sysinline Point3D<bool> operator OP(const Point3D<bool>& A, const Point3D<bool>& B) \
        {return point3D(A.X OP B.X, A.Y OP B.Y, A.Z OP B.Z);} \
    \
    sysinline Point3D<bool> operator OP(const Point3D<bool>& A, bool B) \
        {return point3D(A.X OP B, A.Y OP B, A.Z OP B);} \
    \
    sysinline Point3D<bool> operator OP(bool A, const Point3D<bool>& B) \
        {return point3D(A OP B.X, A OP B.Y, A OP B.Z);}

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

sysinline bool allv(const Point3D<bool>& P)
    {return P.X && P.Y && P.Z;}

sysinline bool anyv(const Point3D<bool>& P)
    {return P.X || P.Y || P.Z;}

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
// ConvertFamilyImpl< Point3D<T> >
//
//================================================================

struct Point3DFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl< Point3D<Type> >
{
    using T = Point3DFamily;
};

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
            return point3D(BaseImpl::func(srcPoint.X), BaseImpl::func(srcPoint.Y), BaseImpl::func(srcPoint.Z));
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
            bool sX = BaseImpl::func(src.X, dst.X);
            bool sY = BaseImpl::func(src.Y, dst.Y);
            bool sZ = BaseImpl::func(src.Z, dst.Z);

            return point3D(sX, sY, sZ);
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
// POINT3D_DEFINE_FUNC1
// POINT3D_DEFINE_FUNC2
// POINT3D_DEFINE_FUNC3
//
//================================================================

#define POINT3D_DEFINE_FUNC1(func) \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& P) \
        {return point3D(func(P.X), func(P.Y), func(P.Z));} \

//----------------------------------------------------------------

#define POINT3D_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Point3D<Type>& B) \
        {return point3D(func(A.X, B.X), func(A.Y, B.Y), func(A.Z, B.Z));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Type& A, const Point3D<Type>& B) \
        {return point3D(func(A, B.X), func(A, B.Y), func(A, B.Z));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Type& B) \
        {return point3D(func(A.X, B), func(A.Y, B), func(A.Z, B));} \

//----------------------------------------------------------------

#define POINT3D_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Point3D<Type>& B, const Point3D<Type>& C) \
        {return point3D(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y), func(A.Z, B.Z, C.Z));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Type& A, const Point3D<Type>& B, const Point3D<Type>& C) \
        {return point3D(func(A, B.X, C.X), func(A, B.Y, C.Y), func(A, B.Z, C.Z));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Type& B, const Point3D<Type>& C) \
        {return point3D(func(A.X, B, C.X), func(A.Y, B, C.Y), func(A.Z, B, C.Z));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Point3D<Type>& B, const Type& C) \
        {return point3D(func(A.X, B.X, C), func(A.Y, B.Y, C), func(A.Z, B.Z, C));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Point3D<Type>& A, const Type& B, const Type& C) \
        {return point3D(func(A.X, B, C), func(A.Y, B, C), func(A.Z, B, C));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Type& A, const Point3D<Type>& B, const Type& C) \
        {return point3D(func(A, B.X, C), func(A, B.Y, C), func(A, B.Z, C));} \
    \
    template <typename Type> \
    sysinline Point3D<Type> func(const Type& A, const Type& B, const Point3D<Type>& C) \
        {return point3D(func(A, B, C.X), func(A, B, C.Y), func(A, B, C.Z));} \

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
// sqrtf
//
//================================================================

POINT3D_DEFINE_FUNC1(absv)
POINT3D_DEFINE_FUNC1(sqrtf)

//================================================================
//
// vectorLengthSq
// vectorLength
//
//================================================================

template <typename Float>
sysinline Float vectorLengthSq(const Point3D<Float>& vec)
    {return square(vec.X) + square(vec.Y) + square(vec.Z);}

//----------------------------------------------------------------

template <typename Float>
sysinline Float vectorLength(const Point3D<Float>& vec)
{
    Float lenSq = vectorLengthSq(vec);
    Float result = fastSqrt(lenSq);
    return result;
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
// scalarProd
//
//================================================================

template <typename Float>
sysinline Float scalarProd(const Point3D<Float>& A, const Point3D<Float>& B)
    {return A.X * B.X + A.Y * B.Y + A.Z * B.Z;}
