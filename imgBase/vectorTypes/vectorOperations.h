#pragma once

#include "numbers/mathIntrinsics.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/float/floatType.h"
#include "vectorTypes/vectorType.h"

//================================================================
//
// float32_x2 binary operations
//
//================================================================

#define TMP_MACRO(OP, ASGOP) \
    \
    sysinline float32_x2 operator OP(const float32_x2& A, const float32_x2& B) \
    { \
        return make_float32_x2 \
        ( \
            A.x OP B.x, \
            A.y OP B.y \
        ); \
    } \
    \
    sysinline float32_x2 operator OP(const float32_x2& A, float32 B) \
    { \
        return make_float32_x2 \
        ( \
            A.x OP B, \
            A.y OP B \
        ); \
    } \
    \
    sysinline float32_x2 operator OP(float32 A, const float32_x2& B) \
    { \
        return make_float32_x2 \
        ( \
            A OP B.x, \
            A OP B.y \
        ); \
    } \
    \
    sysinline float32_x2& operator ASGOP(float32_x2& A, const float32_x2& B) \
    { \
        A.x ASGOP B.x; \
        A.y ASGOP B.y; \
        \
        return A; \
    } \
    \
    sysinline float32_x2& operator ASGOP(float32_x2& A, float32 B) \
    { \
        A.x ASGOP B; \
        A.y ASGOP B; \
        \
        return A; \
    }

TMP_MACRO(*, *=)
TMP_MACRO(/, /=)
TMP_MACRO(+, +=)
TMP_MACRO(-, -=)

#undef TMP_MACRO

//----------------------------------------------------------------

sysinline float32_x2 operator -(const float32_x2& V)
    {return make_float32_x2(-V.x, -V.y);}

sysinline float32_x2 operator +(const float32_x2& V)
    {return V;}

//================================================================
//
// 2X comparisons
//
//================================================================

#define TMP_MACRO(OP) \
    \
    sysinline bool_x2 operator OP(const float32_x2& A, const float32_x2& B) \
    { \
        return make_bool_x2 \
        ( \
            A.x OP B.x, \
            A.y OP B.y \
        ); \
    } \
    \
    sysinline bool_x2 operator OP(const float32_x2& A, float32 B) \
    { \
        return make_bool_x2 \
        ( \
            A.x OP B, \
            A.y OP B \
        ); \
    } \
    \
    sysinline bool_x2 operator OP(float32 A, const float32_x2& B) \
    { \
        return make_bool_x2 \
        ( \
            A OP B.x, \
            A OP B.y \
        ); \
    } \

TMP_MACRO(==)
TMP_MACRO(!=)
TMP_MACRO(>)
TMP_MACRO(<)
TMP_MACRO(>=)
TMP_MACRO(<=)

#undef TMP_MACRO

//================================================================
//
// def
//
//================================================================

template <>
struct DefImpl<float32_x2>
{
    static sysinline bool_x2 func(const float32_x2& value)
        {return make_bool_x2(def(value.x), def(value.y));}
};

//================================================================
//
// 2X bool combining
//
//================================================================

sysinline bool_x2 operator && (const bool_x2& a, const bool_x2& b)
    {return make_bool_x2(a.x && b.x, a.y && b.y);}

sysinline bool_x2 operator || (const bool_x2& a, const bool_x2& b)
    {return make_bool_x2(a.x || b.x, a.y || b.y);}

////

sysinline bool allv(const bool_x2& value)
    {return value.x && value.y;}

sysinline bool anyv(const bool_x2& value)
    {return value.x || value.y;}

//================================================================
//
// VEC2X_DEFINE_FUNC*
//
//================================================================

#define VEC2X_DEFINE_FUNC1(func) \
    \
    sysinline float32_x2 func(const float32_x2& a) \
        {return make_float32_x2(func(a.x), func(a.y));}

////

#define VEC2X_DEFINE_FUNC2(func) \
    \
    sysinline float32_x2 func(const float32_x2& a, const float32_x2& b) \
        {return make_float32_x2(func(a.x, b.x), func(a.y, b.y));}

////

#define VEC2X_DEFINE_FUNC3(func) \
    \
    sysinline float32_x2 func(const float32_x2& a, const float32_x2& b, const float32_x2& c) \
        {return make_float32_x2(func(a.x, b.x, c.x), func(a.y, b.y, c.y));}

////

VEC2X_DEFINE_FUNC2(minv)
VEC2X_DEFINE_FUNC2(maxv)
VEC2X_DEFINE_FUNC2(clampMin)
VEC2X_DEFINE_FUNC2(clampMax)
VEC2X_DEFINE_FUNC3(clampRange)

VEC2X_DEFINE_FUNC1(absv)

//================================================================
//
// float32_x4 binary operations
//
//================================================================

#define TMP_MACRO(OP, ASGOP) \
    \
    sysinline float32_x4 operator OP(const float32_x4& A, const float32_x4& B) \
    { \
        return make_float32_x4 \
        ( \
            A.x OP B.x, \
            A.y OP B.y, \
            A.z OP B.z, \
            A.w OP B.w \
        ); \
    } \
    \
    sysinline float32_x4 operator OP(const float32_x4& A, float32 B) \
    { \
        return make_float32_x4 \
        ( \
            A.x OP B, \
            A.y OP B, \
            A.z OP B, \
            A.w OP B \
        ); \
    } \
    \
    sysinline float32_x4 operator OP(float32 A, const float32_x4& B) \
    { \
        return make_float32_x4 \
        ( \
            A OP B.x, \
            A OP B.y, \
            A OP B.z, \
            A OP B.w \
        ); \
    } \
    \
    sysinline float32_x4& operator ASGOP(float32_x4& A, const float32_x4& B) \
    { \
        A.x ASGOP B.x; \
        A.y ASGOP B.y; \
        A.z ASGOP B.z; \
        A.w ASGOP B.w; \
        \
        return A; \
    } \
    \
    sysinline float32_x4& operator ASGOP(float32_x4& A, float32 B) \
    { \
        A.x ASGOP B; \
        A.y ASGOP B; \
        A.z ASGOP B; \
        A.w ASGOP B; \
        \
        return A; \
    }

TMP_MACRO(*, *=)
TMP_MACRO(/, /=)
TMP_MACRO(+, +=)
TMP_MACRO(-, -=)

#undef TMP_MACRO

//================================================================
//
// 4X unary operations
//
//================================================================

sysinline float32_x4 operator +(const float32_x4& V)
    {return V;}

sysinline float32_x4 operator -(const float32_x4& V)
    {return make_float32_x4(-V.x, -V.y, -V.z, -V.w);}

//================================================================
//
// 4X comparisons
//
//================================================================

#define TMP_MACRO(OP) \
    \
    sysinline bool_x4 operator OP(const float32_x4& A, const float32_x4& B) \
    { \
        return make_bool_x4 \
        ( \
            A.x OP B.x, \
            A.y OP B.y, \
            A.z OP B.z, \
            A.w OP B.w \
        ); \
    } \
    \
    sysinline bool_x4 operator OP(const float32_x4& A, float32 B) \
    { \
        return make_bool_x4 \
        ( \
            A.x OP B, \
            A.y OP B, \
            A.z OP B, \
            A.w OP B \
        ); \
    } \
    \
    sysinline bool_x4 operator OP(float32 A, const float32_x4& B) \
    { \
        return make_bool_x4 \
        ( \
            A OP B.x, \
            A OP B.y, \
            A OP B.z, \
            A OP B.w \
        ); \
    } \

TMP_MACRO(==)
TMP_MACRO(!=)
TMP_MACRO(>)
TMP_MACRO(<)
TMP_MACRO(>=)
TMP_MACRO(<=)

#undef TMP_MACRO

//================================================================
//
// def
//
//================================================================

template <>
struct DefImpl<float32_x4>
{
    static sysinline bool_x4 func(const float32_x4& value)
        {return make_bool_x4(def(value.x), def(value.y), def(value.z), def(value.w));}
};

//================================================================
//
// 4X bool combining
//
//================================================================

sysinline bool_x4 operator && (const bool_x4& a, const bool_x4& b)
    {return make_bool_x4(a.x && b.x, a.y && b.y, a.z && b.z, a.w && b.w);}

sysinline bool_x4 operator || (const bool_x4& a, const bool_x4& b)
    {return make_bool_x4(a.x || b.x, a.y || b.y, a.z || b.z, a.w || b.w);}

////

sysinline bool allv(const bool_x4& value)
    {return value.x && value.y && value.z && value.w;}

sysinline bool anyv(const bool_x4& value)
    {return value.x || value.y || value.z || value.w;}

//================================================================
//
// VEC4X_DEFINE_FUNC*
//
//================================================================

#define VEC4X_DEFINE_FUNC1(func) \
    \
    sysinline float32_x4 func(const float32_x4& a) \
    { \
        return make_float32_x4 \
        ( \
            func(a.x), \
            func(a.y), \
            func(a.z), \
            func(a.w) \
        ); \
    }

#define VEC4X_DEFINE_FUNC2(func) \
    \
    sysinline float32_x4 func(const float32_x4& a, const float32_x4& b) \
    { \
        return make_float32_x4 \
        ( \
            func(a.x, b.x), \
            func(a.y, b.y), \
            func(a.z, b.z), \
            func(a.w, b.w) \
        ); \
    }

#define VEC4X_DEFINE_FUNC3(func) \
    \
    sysinline float32_x4 func(const float32_x4& a, const float32_x4& b, const float32_x4& c) \
    { \
        return make_float32_x4 \
        ( \
            func(a.x, b.x, c.x), \
            func(a.y, b.y, c.y), \
            func(a.z, b.z, c.z), \
            func(a.w, b.w, c.w) \
        ); \
    }

////

VEC4X_DEFINE_FUNC2(minv)
VEC4X_DEFINE_FUNC2(maxv)
VEC4X_DEFINE_FUNC2(clampMin)
VEC4X_DEFINE_FUNC2(clampMax)
VEC4X_DEFINE_FUNC3(clampRange)

VEC4X_DEFINE_FUNC1(absv)

//================================================================
//
// complexConjugate
// complexMul
// scalarProd
//
//================================================================

sysinline float32_x2 complexConjugate(const float32_x2& p)
    {return make_float32_x2(p.x, -p.y);}

//----------------------------------------------------------------

sysinline float32_x2 complexMul(const float32_x2& a, const float32_x2& b)
    {return make_float32_x2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);}

//----------------------------------------------------------------

sysinline float32_x2 complexMad(const float32_x2& sum, const float32_x2& a, const float32_x2& b)
{
    float32_x2 result = sum;

    result.x += a.x * b.x;
    result.x -= a.y * b.y;
    result.y += a.x * b.y;
    result.y += a.y * b.x;

    return result;
}

//----------------------------------------------------------------

sysinline float32 scalarProd(const float32_x2& a, const float32_x2& b)
    {return a.x*b.x + a.y*b.y;}

//================================================================
//
// componentSum
//
//================================================================

sysinline float32 componentSum(const float32_x2& vec)
    {return vec.x + vec.y;}

sysinline float32 componentSum(const float32_x4& vec)
    {return vec.x + vec.y + vec.z + vec.w;}

//================================================================
//
// pi
// 
// (cannot use template constants because of MSVC bugs)
//
//================================================================

#ifndef ROTATION_PI_DEFINED
#define ROTATION_PI_DEFINED

constexpr float32 pi32 = 3.14159265358979324f;
constexpr float64 pi64 = 3.14159265358979324;

#endif

//================================================================
//
// circleCcw
//
//================================================================

template <typename Float>
sysinline auto circleCcw(Float v)
{
    Float angle = v * (2 * Float(pi64));
    Float resultX, resultY;
    nativeCosSin(angle, resultX, resultY);
    return makeVec2(resultX, resultY);
}

//================================================================
//
// vectorLengthSq
//
//================================================================

sysinline float32 vectorLengthSq(const float32_x2& vec)
    {return square(vec.x) + square(vec.y);}

sysinline float32 vectorLengthSq(const float32_x4& vec)
    {return square(vec.x) + square(vec.y) + square(vec.z) + square(vec.w);}

//================================================================
//
// vectorDecompose
//
// Decomposition of a vector to the polar form.
//
//================================================================

sysinline void vectorDecompose(const float32_x2& vec, float32& vectorLengthSq, float32& vectorDivLen, float32& vectorLength, float32_x2& vectorDir)
{
    vectorLengthSq = square(vec.x) + square(vec.y);
    vectorDivLen = recipSqrt(vectorLengthSq);
    vectorLength = vectorLengthSq * vectorDivLen;
    vectorDir = vec * vectorDivLen;

    if (vectorLengthSq == 0)
    {
        vectorLength = 0; 

        vectorDir.x = 1; 
        vectorDir.y = 0;
    }
}

//----------------------------------------------------------------

sysinline void vectorDecompose(const float32_x4& vec, float32& vectorLengthSq, float32& vectorDivLen, float32& vectorLength, float32_x4& vectorDir)
{
    vectorLengthSq = square(vec.x) + square(vec.y) + square(vec.z) + square(vec.w);
    vectorDivLen = recipSqrt(vectorLengthSq);
    vectorLength = vectorLengthSq * vectorDivLen;
    vectorDir = vec * vectorDivLen;

    if (vectorLengthSq == 0)
    {
        vectorLength = 0; 

        vectorDir.x = 1; 
        vectorDir.y = 0; 
        vectorDir.z = 0; 
        vectorDir.w = 0;
    }
}
