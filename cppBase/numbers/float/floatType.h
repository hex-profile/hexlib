#pragma once

#include <cmath>

#include <float.h>
#include <stdlib.h>

#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__))
    #include <xmmintrin.h>
#endif

#include "numbers/float/floatBase.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intType.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Type traits
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// TypeIsBuiltinFloat
// TYPE_IS_BUILTIN_FLOAT
//
//================================================================

template <typename Type>
struct TypeIsBuiltinFloat {static const bool result = false;};

////

#define TMP_MACRO(Type, o) \
    template <> struct TypeIsBuiltinFloat<Type> {static const bool result = true;};

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

////

#define TYPE_IS_BUILTIN_FLOAT(Type) \
    TypeIsBuiltinFloat<Type>::result

//================================================================
//
// TypeIsSignedImpl
//
//================================================================

BUILTIN_FLOAT_FOREACH(TYPE_IS_SIGNED_IMPL, true)

//================================================================
//
// TypeMakeSignedImpl
// TypeMakeUnsignedImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    TYPE_MAKE_SIGNED_IMPL(Type, Type) \
    TYPE_MAKE_UNSIGNED_IMPL(Type, Type)

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// TypeIsControlledImpl
//
//================================================================

BUILTIN_FLOAT_FOREACH(TYPE_IS_CONTROLLED_IMPL, true)

//================================================================
//
// TypeMakeControlledImpl
// TypeMakeUncontrolledImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    TYPE_MAKE_CONTROLLED_IMPL(Type, Type) \
    TYPE_MAKE_UNCONTROLLED_IMPL(Type, Type)

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// TypeMinMaxImpl
//
//================================================================

TYPE_MIN_MAX_IMPL_RUNTIME(float, -FLT_MAX, +FLT_MAX)
TYPE_MIN_MAX_IMPL_RUNTIME(double, -DBL_MAX, +DBL_MAX)

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Controlled type functions
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// NanOfImpl
//
//================================================================

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__CUDA_ARCH__) || defined(__arm__) || defined(__aarch64__)

//----------------------------------------------------------------

COMPILE_ASSERT_EQUAL_LAYOUT(float32, uint32);
static const int32 ieeeFloat32Nan = int32(0xFFC00000UL);

template <>
struct NanOfImpl<float32>
{
    static sysinline float32 func()
    {
    #if defined(__CUDA_ARCH__)
        return __int_as_float(ieeeFloat32Nan);
    #else
        return * (const float32*) &ieeeFloat32Nan;
    #endif
    }
};

sysinline float32 float32Nan()
{
    return NanOfImpl<float32>::func();
}

//----------------------------------------------------------------

COMPILE_ASSERT_EQUAL_LAYOUT(float64, uint64);
static const int64 ieeeFloat64Nan = 0xFFF8000000000000ULL;

//----------------------------------------------------------------

template <>
struct NanOfImpl<float64>
{
    static sysinline float64 func()
    {
    #if defined(__CUDA_ARCH__)
        return __longlong_as_double(ieeeFloat64Nan);
    #else
        return * (const float64*) &ieeeFloat64Nan;
    #endif
    }
};

sysinline float64 float64Nan()
{
    return NanOfImpl<float64>::func();
}

//----------------------------------------------------------------

#else

    #error Implement

#endif

//================================================================
//
// def
//
//================================================================

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__CUDA_ARCH__) || defined(__arm__) || defined(__aarch64__)

//----------------------------------------------------------------

template <>
struct DefImpl<float32>
{
    static sysinline bool func(const float32& value)
    {
    #if defined(__CUDA_ARCH__)
        return isfinite(value);
    #else
        return value >= -FLT_MAX && value <= +FLT_MAX; // Works for NAN
    #endif
    }
};

//----------------------------------------------------------------

template <>
struct DefImpl<float64>
{
    static sysinline bool func(const float64& value)
    {
    #if defined(__CUDA_ARCH__)
        return isfinite(value);
    #else
        return value >= -DBL_MAX && value <= +DBL_MAX; // Works for NAN
    #endif
    }
};

//----------------------------------------------------------------

#else

    #error Implement

#endif

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
// ConvertFamilyImpl<>
//
//================================================================

struct BuiltinFloat;

//----------------------------------------------------------------

BUILTIN_FLOAT_FOREACH(CONVERT_FAMILY_IMPL, BuiltinFloat)

//================================================================
//
// Float -> Float
//
// Always checked.
// Always round nearest.
//
//================================================================

template <ConvertCheck check, ConvertHint hint>
struct ConvertImpl<BuiltinFloat, BuiltinFloat, check, RoundNearest, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& src)
        {
            return Dst(src); // IEEE float checks automatically
        }
    };
};

//----------------------------------------------------------------

template <ConvertHint hint>
struct ConvertImplFlag<BuiltinFloat, BuiltinFloat, RoundNearest, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& src, Dst& dst)
        {
            dst = Dst(src); // IEEE float checks automatically
            return def(dst);
        }
    };
};

//================================================================
//
// Float -> Float
//
// Exact conversion for the same type.
//
//================================================================

template <ConvertCheck check, ConvertHint hint>
struct ConvertImpl<BuiltinFloat, BuiltinFloat, check, RoundExact, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, Dst));

        static sysinline Dst func(const Src& src)
            {return src;}
    };
};

//----------------------------------------------------------------

template <ConvertHint hint>
struct ConvertImplFlag<BuiltinFloat, BuiltinFloat, RoundExact, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, Dst));

        static sysinline bool func(const Src& src, Dst& dst)
            {dst = src; return true;}
    };
};

//================================================================
//
// Int -> Float
//
// Always checked.
// Always round nearest.
//
//================================================================

template <ConvertCheck check, ConvertHint hint>
struct ConvertImpl<BuiltinInt, BuiltinFloat, check, RoundNearest, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& src)
        {
            return Dst(src); // The largest uint fits into the smallest float.
        }
    };
};

//----------------------------------------------------------------

template <ConvertHint hint>
struct ConvertImplFlag<BuiltinInt, BuiltinFloat, RoundNearest, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& src, Dst& dst)
        {
            dst = Dst(src); // The largest uint fits into the smallest float.
            return true;
        }
    };
};

//================================================================
//
// Float -> Int (CUDA GPU)
//
// * Unchecked.
// * Correct work for negative numbers.
//
//================================================================

#if defined(__CUDA_ARCH__)

//----------------------------------------------------------------

template <typename Src, typename Dst, Rounding rounding>
sysinline Dst cudaFloatToInt(Src value);

#define TMP_MACRO(Src, Dst, rounding, baseFunc) \
    template <> \
    sysinline Dst cudaFloatToInt<Src, Dst, rounding>(Src value) \
        {return baseFunc(value);}

TMP_MACRO(float, int, RoundDown, __float2int_rd)
TMP_MACRO(float, int, RoundUp, __float2int_ru)
TMP_MACRO(float, int, RoundNearest, __float2int_rn)

TMP_MACRO(float, unsigned, RoundDown, __float2uint_rd)
TMP_MACRO(float, unsigned, RoundUp, __float2uint_ru)
TMP_MACRO(float, unsigned, RoundNearest, __float2uint_rn)

#undef TMP_MACRO

//----------------------------------------------------------------

template <Rounding rounding, ConvertHint hint>
struct ConvertImpl<BuiltinFloat, BuiltinInt, ConvertUnchecked, rounding, hint>
{
    template <typename Float, typename Int>
    struct Convert
    {
        static sysinline Int func(const Float& src)
            {return cudaFloatToInt<Float, Int, rounding>(src);}
    };
};

//----------------------------------------------------------------

#endif

//================================================================
//
// FLOATTYPE_GENERIC_IEEE_VERSION
//
//================================================================

#if !defined(__CUDA_ARCH__)
    #define FLOATTYPE_GENERIC_IEEE_VERSION 1
#else
    #define FLOATTYPE_GENERIC_IEEE_VERSION 0
#endif

//================================================================
//
// Float -> Int (IEEE FP32)
//
// * Unchecked.
// * Correct for negative numbers.
//
//================================================================

#if FLOATTYPE_GENERIC_IEEE_VERSION

//----------------------------------------------------------------

template <typename Src, typename Dst, Rounding rounding>
sysinline Dst floatToInt(Src value)
    MISSING_FUNCTION_BODY

//----------------------------------------------------------------

template <>
sysinline int32 floatToInt<float32, int32, RoundDown>(float32 src)
{
    // Verified version.
    int32 result = int32(src);

    if (result > src)
        --result;

    return result;
}

//----------------------------------------------------------------

template <>
sysinline uint32 floatToInt<float32, uint32, RoundDown>(float32 src)
{
    return uint32(src);
}

//----------------------------------------------------------------

template <>
sysinline int32 floatToInt<float32, int32, RoundUp>(float32 src)
    {return int32(ceilf(src));}

//----------------------------------------------------------------

template <>
sysinline uint32 floatToInt<float32, uint32, RoundUp>(float32 src)
    {return uint32(ceilf(src));}

//----------------------------------------------------------------

template <>
sysinline int32 floatToInt<float32, int32, RoundNearest>(float32 src)
{

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)

    return _mm_cvtss_si32(_mm_set_ps1(src));

#else

    // Verified 24 bit version, correct rounding for negatives.
    float32 almostHalf = 0.49999997f; // Exactly 0x3EFFFFFF in IEEE 32-bit

    float32 rndVal = src + almostHalf;

    if (src < 0)
        rndVal = src - almostHalf;

    return int32(rndVal);

#endif

}

//----------------------------------------------------------------

template <>
sysinline uint32 floatToInt<float32, uint32, RoundNearest>(float32 src)
{
    // Verified 24 bit version, correct rounding for negatives.
    float32 almostHalf = 0.49999997f; // Exactly 0x3EFFFFFF in IEEE 32-bit
    return uint32(src + almostHalf);
}

//----------------------------------------------------------------

template <Rounding rounding, ConvertHint hint>
struct ConvertImpl<BuiltinFloat, BuiltinInt, ConvertUnchecked, rounding, hint>
{
    template <typename Float, typename Int>
    struct Convert
    {
        COMPILE_ASSERT(sizeof(Int) <= sizeof(int32));
        using PromoInt = TypeSelect<TYPE_IS_SIGNED(Int), int32, uint32>;

        static sysinline Int func(const Float& src)
        {
            return Int(floatToInt<Float, PromoInt, rounding>(src));
        }
    };
};

//----------------------------------------------------------------

#endif

//================================================================
//
// Float -> Int (IEEE FP32)
//
// * Unchecked.
// * Incorrect for negative numbers (fast).
//
//================================================================

#if FLOATTYPE_GENERIC_IEEE_VERSION

//----------------------------------------------------------------

template <>
struct ConvertImpl<BuiltinFloat, BuiltinInt, ConvertUnchecked, RoundDown, ConvertNonneg>
{
    template <typename Float, typename Int>
    struct Convert
    {
        static sysinline Int func(const Float& src)
            {return Int(src);}
    };
};

template <>
struct ConvertImpl<BuiltinFloat, BuiltinInt, ConvertUnchecked, RoundNearest, ConvertNonneg>
{
    template <typename Float, typename Int>
    struct Convert
    {
        static sysinline Int func(const Float& src)
        {
            // Verified 24 bit version
            float32 almostHalf = 0.49999997f; // Exactly 0x3EFFFFFF in IEEE FP32
            return Int(src + almostHalf);
        }
    };
};

//----------------------------------------------------------------

#endif

//================================================================
//
// Float -> Int (IEEE FP32)
//
// * Checked.
// * Correct for negative numbers.
//
//================================================================

#if FLOATTYPE_GENERIC_IEEE_VERSION

//----------------------------------------------------------------

template <typename Float, typename Int, Rounding rounding>
sysinline bool floatFitsInt(Float val)
    MISSING_FUNCTION_BODY

////

#define TMP_MACRO(Float, Int, rounding, minCheck, maxCheck) \
    \
    template <> \
    sysinline bool floatFitsInt<Float, Int, rounding>(Float val) \
        {return minCheck && maxCheck;}

//
// int32/uint32
// Both range ends should be represented in IEEE float32 EXACTLY.
// For example: (val <= 0x7FFFFFFF) is incorrect!
// For such large numbers, all floats are integers.
//

TMP_MACRO(float32, int32, RoundDown, val >= -2147483648.f, val < 2147483648.f)
TMP_MACRO(float32, uint32, RoundDown, val >= 0.f, val < 4294967296.f)

TMP_MACRO(float32, int16, RoundDown, val >= -32768.f, val < 32768.f)
TMP_MACRO(float32, uint16, RoundDown, val >= 0.f, val < 65536.f)

TMP_MACRO(float32, int8, RoundDown, val >= -128.f, val < 128.f)
TMP_MACRO(float32, uint8, RoundDown, val >= 0.f, val < 256.f)

//
//
//

TMP_MACRO(float32, int32, RoundUp, val >= -2147483648.f, val < 2147483648.f)
TMP_MACRO(float32, uint32, RoundUp, val > -1.f, val < 4294967296.f)

TMP_MACRO(float32, int16, RoundUp, val > -32769.f, val <= 32767.f)
TMP_MACRO(float32, uint16, RoundUp, val > -1.f, val <= 65535.f)

TMP_MACRO(float32, int8, RoundUp, val > -129.f, val <= 127.f)
TMP_MACRO(float32, uint8, RoundUp, val > -1.f, val <= 255.f)

//
//
//

TMP_MACRO(float32, int32, RoundNearest, val >= -2147483648.f, val < 2147483648.f)
TMP_MACRO(float32, uint32, RoundNearest, val >= -0.5f, val < 4294967296.f)

TMP_MACRO(float32, int16, RoundNearest, val >= -32768.5f, val < 32767.5f)
TMP_MACRO(float32, uint16, RoundNearest, val >= -0.5f, val < 65535.5f)

TMP_MACRO(float32, int8, RoundNearest, val >= -128.5f, val < 127.5f)
TMP_MACRO(float32, uint8, RoundNearest, val >= -0.5f, val < 255.5f)

////

#undef TMP_MACRO

//----------------------------------------------------------------

template <Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<BuiltinFloat, BuiltinInt, rounding, hint>
{
    template <typename Float, typename Int>
    struct Convert
    {
        using BaseImpl = ConvertImpl<BuiltinFloat, BuiltinInt, ConvertUnchecked, rounding, hint>;
        using BaseConvert = typename BaseImpl::template Convert<Float, Int>;

        static sysinline bool func(const Float& src, Int& dst)
        {
            dst = BaseConvert::func(src);
            return def(src) && floatFitsInt<Float, Int, rounding>(src);
        }
    };
};

//----------------------------------------------------------------

#endif

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

BUILTIN_FLOAT_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)

//================================================================
//
// minv
// maxv
//
//================================================================

#define TMP_MACRO(Type, baseMin, baseMax) \
    \
    template <> \
    sysinline Type minv(const Type& A, const Type& B) \
        {return baseMin(A, B);} \
    \
    template <> \
    sysinline Type maxv(const Type& A, const Type& B) \
        {return baseMax(A, B);} \
    \
    template <> \
    sysinline Type clampMin(const Type& X, const Type& minValue) \
        {return baseMax(X, minValue);} \
    \
    template <> \
    sysinline Type clampMax(const Type& X, const Type& maxValue) \
        {return baseMin(X, maxValue);} \
    \
    template <> \
    sysinline Type clampRange(const Type& X, const Type& minValue, const Type& maxValue) \
        {return baseMin(baseMax(X, minValue), maxValue);} \

////

#if defined(__CUDA_ARCH__)

    TMP_MACRO(float32, fminf, fmaxf)
    TMP_MACRO(float64, fmin, fmax)

#elif defined(_MSC_VER)

    // slow
    TMP_MACRO(float32, __min, __max)
    TMP_MACRO(float64, __min, __max)

#elif defined(__GNUC__)

    TMP_MACRO(float32, fminf, fmaxf)
    TMP_MACRO(float64, fmin, fmax)

#else

    #error Need to implement

#endif

////

#undef TMP_MACRO

//================================================================
//
// convertFloat32
// convertFloat64
//
//================================================================

template <typename Src>
sysinline auto convertFloat32(const Src& src)
    {return convertNearest<float32>(src);}

template <typename Src>
sysinline auto convertFloat64(const Src& src)
    {return convertNearest<float64>(src);}

//================================================================
//
// absv
// floorv
// ceilv
//
//================================================================

#define TMP_MACRO(func, Type, baseFunc) \
    \
    template <> \
    sysinline Type func(const Type& value) \
        {return baseFunc(value);}

TMP_MACRO(absv, float, fabsf)
TMP_MACRO(absv, double, fabs)

TMP_MACRO(floorv, float, floorf)
TMP_MACRO(floorv, double, floor)

TMP_MACRO(ceilv, float, ceilf)
TMP_MACRO(ceilv, double, ceil)

#undef TMP_MACRO

//================================================================
//
// scalarProd
//
//================================================================

template <>
sysinline float32 scalarProd(const float32& A, const float32& B)
    {return A * B;}

template <>
sysinline float64 scalarProd(const float64& A, const float64& B)
    {return A * B;}

//================================================================
//
// vectorSum
//
//================================================================

template <>
sysinline float32 vectorSum(const float32& vec)
    {return vec;}

template <>
sysinline float64 vectorSum(const float64& vec)
    {return vec;}
