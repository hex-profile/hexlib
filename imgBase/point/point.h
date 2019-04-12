#pragma once

#include "point/pointBase.h"
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
// VectorBaseImpl<Point>
//
//================================================================

template <typename Type>
struct VectorBaseImpl<Point<Type>>
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<Point>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl<Point<OldBase>, NewBase>
{
    using T = ::Point<NewBase>;
};

//================================================================
//
// VectorExtendImpl<Point>
//
//================================================================

template <typename Type>
struct VectorExtendImpl<Point<Type>>
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
        return Point<bool>
        {
            def(value.X), 
            def(value.Y)
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
sysinline auto operator +(const Point<Type>& P)
    {return P;}

template <typename Type>
sysinline auto operator -(const Point<Type>& P)
    {return point(-P.X, -P.Y);}

template <typename Type>
sysinline auto operator !(const Point<Type>& P)
    {return point(!P.X, !P.Y);}

//================================================================
//
// Binary operators.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point<TypeA>& A, const Point<TypeB>& B) \
        {return point(A.X OP B.X, A.Y OP B.Y);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const Point<TypeA>& A, const TypeB& B) \
        {return point(A.X OP B, A.Y OP B);} \
    \
    template <typename TypeA, typename TypeB> \
    sysinline auto operator OP(const TypeA& A, const Point<TypeB>& B) \
        {return point(A OP B.X, A OP B.Y);}

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
    sysinline auto& operator OP(Point<TypeA>& A, const Point<TypeB>& B) \
    { \
        A.X OP B.X; \
        A.Y OP B.Y; \
        return A; \
    } \
    template <typename TypeA, typename TypeB> \
    sysinline auto& operator OP(Point<TypeA>& A, const TypeB& B) \
    { \
        A.X OP B; \
        A.Y OP B; \
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
sysinline bool allv(const Point<Type>& P)
    {return allv(P.X) && allv(P.Y);}

template <typename Type>
sysinline bool anyv(const Point<Type>& P)
    {return anyv(P.X) || anyv(P.Y);}

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
// ConvertFamilyImpl<Point<T>>
//
//================================================================

struct PointFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl<Point<Type>>
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
            return point
            (
                BaseImpl::func(srcPoint.X), 
                BaseImpl::func(srcPoint.Y)
            );
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
            return point
            (
                BaseImpl::func(src.X, dst.X),
                BaseImpl::func(src.Y, dst.Y)
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
sysinline void exchange(Point<Type>& A, Point<Type>& B)
{
    exchange(A.X, B.X);
    exchange(A.Y, B.Y);
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
    sysinline auto func(const Point<Type>& P) \
        {return point(func(P.X), func(P.Y));} 

//----------------------------------------------------------------

#define POINT_DEFINE_FUNC2(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Point<Type>& B) \
        {return point(func(A.X, B.X), func(A.Y, B.Y));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point<Type>& B) \
        {return point(func(A, B.X), func(A, B.Y));} \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Type& B) \
        {return point(func(A.X, B), func(A.Y, B));}

//----------------------------------------------------------------

#define POINT_DEFINE_FUNC3(func) \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Point<Type>& B, const Point<Type>& C) \
        {return point(func(A.X, B.X, C.X), func(A.Y, B.Y, C.Y));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point<Type>& B, const Point<Type>& C) \
        {return point(func(A, B.X, C.X), func(A, B.Y, C.Y));} \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Type& B, const Point<Type>& C) \
        {return point(func(A.X, B, C.X), func(A.Y, B, C.Y));} \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Point<Type>& B, const Type& C) \
        {return point(func(A.X, B.X, C), func(A.Y, B.Y, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Point<Type>& A, const Type& B, const Type& C) \
        {return point(func(A.X, B, C), func(A.Y, B, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Point<Type>& B, const Type& C) \
        {return point(func(A, B.X, C), func(A, B.Y, C));} \
    \
    template <typename Type> \
    sysinline auto func(const Type& A, const Type& B, const Point<Type>& C) \
        {return point(func(A, B, C.X), func(A, B, C.Y));}

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
//
//================================================================

POINT_DEFINE_FUNC1(floorf)
POINT_DEFINE_FUNC1(ceilf)
POINT_DEFINE_FUNC1(absv)
