#pragma once

#include <stdlib.h>
#include <limits.h>

#include "numbers/int/intBase.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intConvertChk.h"
#include "prepTools/prepArg.h"

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
// TypeIsBuiltinInt
// TYPE_IS_BUILTIN_INT
//
//================================================================

template <typename Type>
struct TypeIsBuiltinInt {static const bool result = false;};

////

#define TMP_MACRO(Type, o) \
    template <> struct TypeIsBuiltinInt<Type> {static const bool result = true;};

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

////

#define TYPE_IS_BUILTIN_INT(Type) \
    TypeIsBuiltinInt< Type >::result

//================================================================
//
// TypeIsSignedImpl
//
//================================================================

TYPE_IS_SIGNED_IMPL(bool, false)

TYPE_IS_SIGNED_IMPL(char, (CHAR_MIN < 0))
TYPE_IS_SIGNED_IMPL(signed char, true)
TYPE_IS_SIGNED_IMPL(unsigned char, false)

TYPE_IS_SIGNED_IMPL(signed short, true)
TYPE_IS_SIGNED_IMPL(unsigned short, false)

TYPE_IS_SIGNED_IMPL(signed int, true)
TYPE_IS_SIGNED_IMPL(unsigned int, false)

TYPE_IS_SIGNED_IMPL(signed long, true)
TYPE_IS_SIGNED_IMPL(unsigned long, false)

TYPE_IS_SIGNED_IMPL(signed long long, true)
TYPE_IS_SIGNED_IMPL(unsigned long long, false)

//================================================================
//
// TypeMakeSignedImpl
// TypeMakeUnsignedImpl
//
//================================================================

#define TMP_MACRO(SignedType, UnsignedType) \
    \
    TYPE_MAKE_SIGNED_IMPL(SignedType, SignedType) \
    TYPE_MAKE_SIGNED_IMPL(UnsignedType, SignedType) \
    TYPE_MAKE_UNSIGNED_IMPL(SignedType, UnsignedType) \
    TYPE_MAKE_UNSIGNED_IMPL(UnsignedType, UnsignedType)

TMP_MACRO(signed char, unsigned char)
TMP_MACRO(signed short, unsigned short)
TMP_MACRO(signed int, unsigned int)
TMP_MACRO(signed long, unsigned long)
TMP_MACRO(signed long long, unsigned long long)

#undef TMP_MACRO

////

TYPE_MAKE_SIGNED_IMPL(char, signed char)
TYPE_MAKE_UNSIGNED_IMPL(char, unsigned char)

//================================================================
//
// TypeIsControlledImpl
//
//================================================================

BUILTIN_INT_FOREACH(TYPE_IS_CONTROLLED_IMPL, false)

//================================================================
//
// TypeMakeUncontrolledImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    TYPE_MAKE_UNCONTROLLED_IMPL(Type, Type)

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// Type min and max.
//
//================================================================

TYPE_MIN_MAX_IMPL_BOTH(bool, false, true)

TYPE_MIN_MAX_IMPL_BOTH(char, CHAR_MIN, CHAR_MAX)
TYPE_MIN_MAX_IMPL_BOTH(signed char, SCHAR_MIN, SCHAR_MAX)
TYPE_MIN_MAX_IMPL_BOTH(unsigned char, 0, UCHAR_MAX)

TYPE_MIN_MAX_IMPL_BOTH(signed short, SHRT_MIN, SHRT_MAX)
TYPE_MIN_MAX_IMPL_BOTH(unsigned short, 0, USHRT_MAX)

TYPE_MIN_MAX_IMPL_BOTH(signed int, INT_MIN, INT_MAX)
TYPE_MIN_MAX_IMPL_BOTH(unsigned int, 0, UINT_MAX)

TYPE_MIN_MAX_IMPL_BOTH(signed long, LONG_MIN, LONG_MAX)
TYPE_MIN_MAX_IMPL_BOTH(unsigned long, 0, ULONG_MAX)

TYPE_MIN_MAX_IMPL_BOTH(signed long long, LLONG_MIN, LLONG_MAX)
TYPE_MIN_MAX_IMPL_BOTH(unsigned long long, 0, ULLONG_MAX)

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
// def
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    struct DefImpl<Type> \
    { \
        static sysinline bool func(const Type& value) \
            {return true;} \
    };

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// nanOfImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    sysinline Type nanOfImpl() \
        {return TYPE_MIN(Type);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

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

struct BuiltinInt;

//----------------------------------------------------------------

BUILTIN_INT_FOREACH(CONVERT_FAMILY_IMPL, BuiltinInt)

//================================================================
//
// bool -> Int
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact
struct ConvertImpl<bool, BuiltinInt, ConvertUnchecked, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& value)
        {
            return Dst(value);
        }
    };
};

//----------------------------------------------------------------

template <Rounding rounding, ConvertHint hint> // exact
struct ConvertImplFlag<bool, BuiltinInt, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& value, Dst& dst)
        {
            dst = Dst(value);
            return true;
        }
    };
};

//================================================================
//
// Int -> Int (unchecked)
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact
struct ConvertImpl<BuiltinInt, BuiltinInt, ConvertUnchecked, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& value)
        {
            return Dst(value); // exact
        }
    };
};

//================================================================
//
// Int -> Int (checked)
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact
struct ConvertImplFlag<BuiltinInt, BuiltinInt, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& src, Dst& dst)
        {
            dst = Dst(src); // exact
            return intConvertChk::Fit<Src, Dst>::Code::func(src);
        }
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

BUILTIN_INT_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)

//================================================================
//
// minv
// maxv
//
//================================================================

#define TMP_DEFINE_INT_MINMAX(Type, baseMin, baseMax) \
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

#define TMP_DEFINE_INT_HELPER(Type, par) \
    TMP_DEFINE_INT_MINMAX(Type, PREP_ARG2_0 par, PREP_ARG2_1 par)

////

#if defined(__CUDA_ARCH__)

    BUILTIN_INT_FOREACH(TMP_DEFINE_INT_HELPER, (min, max))

#elif defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)

    // slow
    template <typename T> sysinline T genericMin(T a, T b) {return a < b ? a : b;}
    template <typename T> sysinline T genericMax(T a, T b) {return a > b ? a : b;}
    BUILTIN_INT_FOREACH(TMP_DEFINE_INT_HELPER, (genericMin, genericMax))

#elif defined(__arm__) || defined(__aarch64__)

    // slow
    template <typename T> sysinline T genericMin(T a, T b) {return a < b ? a : b;}
    template <typename T> sysinline T genericMax(T a, T b) {return a > b ? a : b;}
    BUILTIN_INT_FOREACH(TMP_DEFINE_INT_HELPER, (genericMin, genericMax))

#else

    #error Need to implement

#endif

////

#undef TMP_DEFINE_INT_HELPER
#undef TMP_DEFINE_INT_MINMAX

//================================================================
//
// absv
//
//================================================================

#define TMP_MACRO(Type, baseFunc) \
    \
    template <> \
    sysinline Type absv(const Type& value) \
        {return (Type) baseFunc(value);}

TMP_MACRO(signed char, abs)
TMP_MACRO(signed short, abs)
TMP_MACRO(signed int, abs)
TMP_MACRO(signed long, labs)

#undef TMP_MACRO

//================================================================
//
// isPower2
//
//================================================================

template <typename Type>
sysinline bool isPowerTwoImpl(const Type& value)
    {return COMPILE_IS_POWER2(value);}

template <>
sysinline bool isPowerTwoImpl(const bool& value)
    {return value;}

//----------------------------------------------------------------

#define TMP_MACRO(Type, o) \
    template <> \
    sysinline bool isPower2(const Type& value) \
        {return isPowerTwoImpl(value);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
