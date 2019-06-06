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
// nativeRecip
//
// Should give ALMOST full precision
//
//================================================================

template <typename Type>
Type nativeRecip(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDACC__)

    template <>
    sysinline float32 nativeRecip(const float32& value)
        {return __fdividef(1, value);} // 6 instructions on Kepler

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 nativeRecip(const float32& value)
        {return 1.f / value;}

#else

    #error

#endif

//================================================================
//
// nativeRecipZero
//
// In case of zero input, returns zero.
//
//================================================================

template <typename Type>
sysinline Type nativeRecipZero(const Type& value)
{
    Type result = nativeRecip(value);
    if_not (def(result)) result = 0;
    return result;
}

//================================================================
//
// nativeDivide
//
// Should give ALMOST full precision
//
//================================================================

template <typename Type>
Type nativeDivide(const Type& A, const Type& B);

//----------------------------------------------------------------

#if defined(__CUDACC__)

    template <>
    sysinline float32 nativeDivide(const float32& A, const float32& B)
        {return __fdividef(A, B);}

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 nativeDivide(const float32& A, const float32& B)
        {return A / B;}

#else

    #error

#endif

//================================================================
//
// recipSqrt
//
// Should give ALMOST full precision.
//
// fastSqrt
//
// Does not handle +inf.
// Should give ALMOST full precision.
//
//================================================================

template <typename Type>
sysinline Type recipSqrt(const Type& value);

template <typename Type>
sysinline Type fastSqrt(const Type& value);

//================================================================
//
// recipSqrt
// fastSqrt
//
//================================================================

#if defined(__CUDACC__)

    template <>
    sysinline float32 recipSqrt(const float32& value)
    {
        return rsqrtf(value);
    }

    template <>
    sysinline float32 fastSqrt(const float32& x)
    {
        float32 result = x * rsqrtf(x);
        if (x == 0) result = 0;
        return result;
    }

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    sysinline float32 recipSqrt(float32 x)
        {return 1.f / sqrtf(x);}

    template <>
    sysinline float32 fastSqrt(const float32& x)
        {return sqrtf(x);}

#else

    #error

#endif

//================================================================
//
// nativeLog2
//
// Should give ALMOST full precision.
//
//================================================================

template <typename Type>
sysinline Type nativeLog2(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline float32 nativeLog2(const float32& value)
        {return __log2f(value);}

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)

    template <>
    sysinline float32 nativeLog2(const float32& value)
        {return logf(value) * 1.44269504088896341f;}

#else

    #error

#endif

//================================================================
//
// nativePow2
//
// Should give ALMOST full precision.
//
//================================================================

template <typename Type>
sysinline Type nativePow2(const Type& value);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline float32 nativePow2(const float32& value)
        {return exp2f(value);}

#else

    template <>
    sysinline float32 nativePow2(const float32& value)
        {return expf(0.6931471805599453f * value);}

#endif

//================================================================
//
// nativeCosSin
//
//================================================================

template <typename Float>
sysinline void nativeCosSin(Float angle, Float& rX, Float& rY);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline void nativeCosSin(float32 angle, float32& rX, float32& rY)
        {__sincosf(angle, &rY, &rX);}

    template <>
    sysinline void nativeCosSin(float64 angle, float64& rX, float64& rY)
        {sincos(angle, &rY, &rX);}

#elif defined(__GNUC__)

    template <>
    sysinline Point<float32> nativeCosSin(float32 angle, float32& rX, float32& rY)
        {sincosf(angle, &rY, &rX);}

    template <>
    sysinline Point<float64> nativeCosSin(float64 angle, float64& rX, float64& rY)
        {sincos(angle, &rY, &rX);}

#else

    template <>
    sysinline void nativeCosSin(float32 angle, float32& rX, float32& rY)
        {rX = cosf(angle); rY = sinf(angle);}

    template <>
    sysinline void nativeCosSin(float64 angle, float64& rX, float64& rY)
        {rX = cos(angle); rY = sin(angle);}

#endif

//================================================================
//
// nativeAtan2
//
// Returns value in range [-Pi, +Pi].
//
//================================================================

template <typename Float>
sysinline Float nativeAtan2(Float rY, Float rX);

template <>
sysinline float32 nativeAtan2(float32 rY, float32 rX)
    {return atan2f(rY, rX);}

template <>
sysinline float64 nativeAtan2(float64 rY, float64 rX)
    {return atan2(rY, rX);}
