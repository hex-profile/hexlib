#pragma once

#include <limits.h>

#include "compileTools/typeOf.h"
#include "compileTools/compileTools.h"

//================================================================
//
// Tools for getting the result type of a binary or unary operation.
//
//================================================================

//================================================================
//
// typeopCreateType
//
//================================================================

template <typename Type>
Type& typeopCreateType();

//================================================================
//
// TYPEOP_BINARY
// TYPEOP_PREFIX
// TYPEOP_POSTFIX
//
// Return the type of binary or unary operation.
//
// Usage:
// TYPEOP_BINARY(char, +, short)
// TYPEOP_PREFIX(-, long)
// etc.
//
//================================================================

#define TYPEOP_BINARY(T1, OP, T2) \
    typename TYPEOF(typeopCreateType< T1 >() OP typeopCreateType< T2 >())

#define TYPEOP_PREFIX(OP, T) \
    typename TYPEOF(OP typeopCreateType< T >())

#define TYPEOP_POSTFIX(T, OP) \
    typename TYPEOF(typeopCreateType< T >() OP)

//================================================================
//
// TYPEOP_BINARY_PREDICT(T1, T2)
// TYPEOP_UNARY_PREDICT(T)
//
//----------------------------------------------------------------
//
// The above approach doesn't work in template functions
// when compiler matches a template function by argument types.
//
// This the below tool was created, which allows to "predict"
// type of binary or unary operation.
//
// The result is NOT guarenteed and should be checked with COMPILE_ASSERT
// for exact match when it's possble (in the function body).
//
//================================================================

//================================================================
//
// TypeIntegralPromotion
//
// Prior integer type promotion, according to the C++ standard.
//
//================================================================

template <typename Type>
struct TypeIntegralPromotion {using T = Type;};

#define TMP_MACRO(Src, Dst) \
    template <> struct TypeIntegralPromotion<Src> {using T = Dst;};

////

TMP_MACRO(bool, int)

TMP_MACRO(char, int)
TMP_MACRO(signed char, int)
TMP_MACRO(unsigned char, int)

TMP_MACRO(signed short, int)
TMP_MACRO(unsigned short, int)

////

#undef TMP_MACRO

//================================================================
//
// TypeIntegralConversionRank
//
// According to the C++ standard: for equal rank, unsigned type wins.
//
//================================================================

template <typename Type>
struct TypeIntegralConversionRank {static const int value = 0x7FFF;};

#define TMP_MACRO(Type, rank) \
    template <> struct TypeIntegralConversionRank<Type> {static const int value = (rank);};

////

TMP_MACRO(bool, 0)

TMP_MACRO(signed char, 1)
TMP_MACRO(unsigned char, 2)
TMP_MACRO(char, CHAR_MIN < 0 ? 1 : 2)

TMP_MACRO(signed short, 3)
TMP_MACRO(unsigned short, 4)

TMP_MACRO(signed int, 5)
TMP_MACRO(unsigned int, 6)

TMP_MACRO(signed long int, 7)
TMP_MACRO(unsigned long int, 8)

TMP_MACRO(signed long long int, 9)
TMP_MACRO(unsigned long long int, 10)

TMP_MACRO(float, 11)
TMP_MACRO(double, 12)
TMP_MACRO(long double, 13)

////

#undef TMP_MACRO

//================================================================
//
// TypeOpBinaryPredict
//
//================================================================

template <typename T1, typename T2>
struct TypeOpBinaryPredict
{
    using expT1 = typename TypeIntegralPromotion<T1>::T;
    using expT2 = typename TypeIntegralPromotion<T2>::T;

    static const int rankT1 = TypeIntegralConversionRank<expT1>::value;
    static const int rankT2 = TypeIntegralConversionRank<expT2>::value;

    using T = TYPE_SELECT(rankT2 > rankT1, expT2, expT1);
};

//----------------------------------------------------------------

#define TYPEOP_BINARY_PREDICT(T1, T2) \
    typename TypeOpBinaryPredict<TYPE_CLEANSE(T1), TYPE_CLEANSE(T2)>::T

#define TYPEOP_UNARY_PREDICT(Type) \
    typename TypeIntegralPromotion<TYPE_CLEANSE(Type)>::T
