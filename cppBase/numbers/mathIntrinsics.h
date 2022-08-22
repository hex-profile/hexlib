#pragma once

#include <cmath>

#include "numbers/float/floatType.h"

//================================================================
//
// saturate
//
// Clamps a value to [0, 1] range.
//
//================================================================

#if defined(__CUDA_ARCH__)

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    sysinline float32 saturate(float32 value)
        {return clampRange(value, 0.f, 1.f);}

#else

    #error

#endif

//================================================================
//
// linerp
// linearIf
//
//================================================================

template <typename Selector, typename Value>
sysinline Value linerp(const Selector& alpha, const Value& falseValue, const Value& trueValue)
{
    return falseValue + alpha * (trueValue - falseValue);
}

template <typename Selector, typename Value>
sysinline Value linearIf(const Selector& alpha, const Value& trueValue, const Value& falseValue)
{
    return falseValue + alpha * (trueValue - falseValue);
}

//================================================================
//
// fastRecip
//
// Should give ALMOST full precision.
//
// Generates 6 instructions on Pascal.
//
//================================================================

template <typename Type>
Type fastRecip(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDACC__)

    template <>
    sysinline float32 fastRecip(const float32& value)
        {return __fdividef(1, value);} // 6 instructions on Kepler, do not change to __frcp_rn, it is much slower!

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 fastRecip(const float32& value)
        {return 1 / value;}

    template <>
    sysinline float64 fastRecip(const float64& value)
        {return 1 / value;}

#else

    #error

#endif

//================================================================
//
// fastRecipZero
//
// Should give ALMOST full precision.
// In case of zero input, returns zero.
//
// Generates 8 instructions on Pascal.
//
//================================================================

template <typename Type>
sysinline Type fastRecipZero(const Type& value)
{
    Type result = fastRecip(value);
    if_not (def(result)) result = 0;
    return result;
}

//================================================================
//
// fastDivide
//
// Should give ALMOST full precision.
//
// Generates 5 instructions on Pascal.
//
//================================================================

template <typename Type>
Type fastDivide(const Type& A, const Type& B);

//----------------------------------------------------------------

#if defined(__CUDACC__)

    template <>
    sysinline float32 fastDivide(const float32& A, const float32& B)
        {return __fdividef(A, B);}

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 fastDivide(const float32& A, const float32& B)
        {return A / B;}

#else

    #error

#endif

//================================================================
//
// fastSqrt
// Should give ALMOST full precision.
// Generates 7 instructions on Pascal.
//
// recipSqrt
// Should give ALMOST full precision.
// Generates 4 instructions on Pascal.
//
//================================================================

#if defined(__CUDACC__)

    template <>
    sysinline float32 recipSqrt(const float32& value)
    {
        return rsqrtf(value); // Do not change to __frsqrt_rn, it is much slower!
    }

    template <>
    sysinline float32 fastSqrt(const float32& value)
    {
        // Do not change to "sqrtf", it is much slower!
        float32 result = value * rsqrtf(value);
        if (value == 0) result = 0;
        return result;
    }

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 recipSqrt(const float32& value)
        {return 1 / sqrtf(value);}

    template <>
    sysinline float32 fastSqrt(const float32& value)
        {return sqrtf(value);}

    ////

    template <>
    sysinline float64 recipSqrt(const float64& value)
        {return 1 / sqrt(value);}

    template <>
    sysinline float64 fastSqrt(const float64& value)
        {return sqrt(value);}

#else

    #error

#endif

//================================================================
//
// fastLog2
//
// Should give ALMOST full precision.
// Generates 4 instructions on Pascal.
//
//================================================================

template <typename Type>
sysinline Type fastLog2(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline float32 fastLog2(const float32& value)
        {return __log2f(value);}

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 fastLog2(const float32& value)
        {return logf(value) * 1.44269504088896341f;}

    template <>
    sysinline float64 fastLog2(const float64& value)
        {return log(value) * 1.44269504088896341;}

#else

    #error

#endif

//================================================================
//
// fastPow2
//
// Should give ALMOST full precision.
// Generates 5 instructions on Pascal.
//
//================================================================

template <typename Type>
sysinline Type fastPow2(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline float32 fastPow2(const float32& value)
        {return exp2f(value);}

#else

    template <>
    sysinline float32 fastPow2(const float32& value)
        {return expf(0.6931471805599453f * value);}

#endif

//================================================================
//
// fastCosSin
//
// Should give ALMOST full precision.
// Generates 3 instructions on Pascal.
//
//================================================================

template <typename Float>
sysinline void fastCosSin(Float angle, Float& rX, Float& rY);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline void fastCosSin(float32 angle, float32& rX, float32& rY)
        {__sincosf(angle, &rY, &rX);}

    template <>
    sysinline void fastCosSin(float64 angle, float64& rX, float64& rY)
        {sincos(angle, &rY, &rX);}

#elif defined(__GNUC__)

    template <>
    sysinline void fastCosSin(float32 angle, float32& rX, float32& rY)
        {sincosf(angle, &rY, &rX);}

    template <>
    sysinline void fastCosSin(float64 angle, float64& rX, float64& rY)
        {sincos(angle, &rY, &rX);}

#else

    template <>
    sysinline void fastCosSin(float32 angle, float32& rX, float32& rY)
        {rX = cosf(angle); rY = sinf(angle);}

    template <>
    sysinline void fastCosSin(float64 angle, float64& rX, float64& rY)
        {rX = cos(angle); rY = sin(angle);}

#endif

//================================================================
//
// exactAtan2
//
// Returns value in range [-Pi, +Pi].
//
// Maximal precision, but slow!
//
//================================================================

template <typename Float>
sysinline Float exactAtan2(Float rY, Float rX);

template <>
sysinline float32 exactAtan2(float32 rY, float32 rX)
    {return atan2f(rY, rX);}

template <>
sysinline float64 exactAtan2(float64 rY, float64 rX)
    {return atan2(rY, rX);}

//================================================================
//
// ldexpv
//
//================================================================

template <typename Float>
sysinline Float ldexpv(Float value, int exp);

template <>
sysinline float32 ldexpv(float32 value, int exp)
    {return ldexpf(value, exp);}

template <>
sysinline float64 ldexpv(float64 value, int exp)
    {return ldexp(value, exp);}
