#pragma once

#if defined(_MSC_VER)
    #define _DO_NOT_DECLARE_INTERLOCKED_INTRINSICS_IN_MEMORY
    #include <intrin.h>
    #define EMUL_AVAILABLE
#endif

#include "numbers/interface/numberInterface.h"
#include "numbers/int/intType.h"

namespace safeint32 {

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Basic definitions
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// Base
//
// Corresponding built-in type.
//
//================================================================

using Base = int32;
using BaseUnsigned = uint32;

#define INT32C__BITCOUNT 32

//================================================================
//
// Constants
//
//================================================================

COMPILE_ASSERT(INT32C__BITCOUNT == 32);

static const Base indefinite = -0x7FFFFFFF-1;
static const Base valMin = -0x7FFFFFFF;
static const Base valMax = +0x7FFFFFFF;

static const bool rangeIsSymmetric = true;

//================================================================
//
// Class
//
// a class to control access
//
//================================================================

class Class
{

public:

    sysinline friend Base access(const Class& V);
    sysinline friend Class create(Base V);

private:

    Base _;

};

sysinline Base access(const Class& value)
{
    return value._;
}

sysinline Class create(Base value)
{
    Class result;
    result._ = value;
    return result;
}

//================================================================
//
// Type selection
//
//================================================================

#if defined(_DEBUG) && !defined(__CUDA_ARCH__)

    using Type = Class;

    #define INT32C__CREATE(value) safeint32::create(value)
    #define INT32C__ACCESS(value) safeint32::access(value)

#else

    enum Type // to be passed in registers
    {
        enumAbsMax = +0x7FFFFFFF,
        enumIndefinite = -0x7FFFFFFF-1
    };

    COMPILE_ASSERT(Type(-1) < 0); // Is signed type
    COMPILE_ASSERT(sizeof(Type) == sizeof(Base));

    #define INT32C__CREATE(value) safeint32::Type(value)
    #define INT32C__ACCESS(value) safeint32::Base(value)

#endif

//================================================================
//
// INT32C__INDEFINITE
//
//================================================================

#define INT32C__INDEFINITE \
    (int32(-0x7FFFFFFF) - 1)

//----------------------------------------------------------------

}

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
// TypeIsSignedImpl
// TypeMakeSignedImpl
//
//================================================================

TYPE_IS_SIGNED_IMPL(safeint32::Type, true)

TYPE_MAKE_SIGNED_IMPL(safeint32::Type, safeint32::Type)

//================================================================
//
// TypeIsControlledImpl
//
// TypeMakeControlledImpl
// TypeMakeUncontrolledImpl
//
//================================================================

TYPE_IS_CONTROLLED_IMPL(safeint32::Type, true)

TYPE_MAKE_CONTROLLED_IMPL(safeint32::Type, safeint32::Type)
TYPE_MAKE_UNCONTROLLED_IMPL(safeint32::Type, safeint32::Base)
TYPE_MAKE_CONTROLLED_IMPL(safeint32::Base, safeint32::Type)

//================================================================
//
// typeMin / typeMax
//
//================================================================

TYPE_MIN_MAX_IMPL_RUNTIME(safeint32::Type, INT32C__CREATE(safeint32::valMin), INT32C__CREATE(safeint32::valMax))

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Basic arithmetic operations
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// Unary + and -
//
//================================================================

sysinline safeint32::Type operator +(const safeint32::Type& V)
    {return V;}

//----------------------------------------------------------------

COMPILE_ASSERT(safeint32::indefinite == safeint32::Base(0x80000000L));

sysinline safeint32::Type operator -(safeint32::Type V)
{
    // On this platform, applying unary minus to 32-bit value 0x80000000
    // gives 0x80000000, that is NAN; so no other actions required

    return INT32C__CREATE(-INT32C__ACCESS(V));
}

//================================================================
//
// safeint32OperatorAdd
//
// C++ version compiles to ~15 instructions.
//
//----------------------------------------------------------------
//
// Original version:
//
// @safeint32OperatorAdd@8 proc
//
//         cmp     ecx, INT32C__INDEFINITE
//         je      nanExit
//         cmp     edx, INT32C__INDEFINITE
//         je      nanExit
//         mov     eax, ecx
//         add     eax, edx ; either OF will be set, or the result is already 0x80000000
//         jo      nanExit
//         ret
//
// nanExit:
//         mov     eax, INT32C__INDEFINITE
//         ret
//
// @safeint32OperatorAdd@8 endp
//
//================================================================

template <>
struct SafeBinaryImpl<safeint32::Base, OpAdd>
{
    struct Code
    {
        static sysinline bool func(safeint32::Base X, safeint32::Base Y, safeint32::Base& R)
        {
            //
            // If the numbers have different sign, overflow is impossible.
            // If the numbers have equal sign, but the result has different sign, we have overflow.
            //

            R = X + Y;
            int32 ovf = ~(X ^ Y) & (R ^ X); // MSB is overflow flag

            return ovf >= 0;
        }
    };
};

//----------------------------------------------------------------

sysinline safeint32::Base safeint32OperatorAdd(safeint32::Base X, safeint32::Base Y)
{
    safeint32::Base R = 0;

    if (X == INT32C__INDEFINITE)
        goto nanExit;

    if (Y == INT32C__INDEFINITE)
        goto nanExit;

    if_not ((SafeBinaryImpl<safeint32::Base, OpAdd>::Code::func(X, Y, R)))
        goto nanExit;

    return R;

nanExit:

    return INT32C__INDEFINITE;
}

//================================================================
//
// safeint32OperatorSub
//
// C++ version compiles to ~15 instructions.
//
//----------------------------------------------------------------
//
// Original version:
//
// @safeint32OperatorSub@8 proc
//
//         cmp     ecx, INT32C__INDEFINITE
//         je      nanExit
//         cmp     edx, INT32C__INDEFINITE
//         je      nanExit
//         mov     eax, ecx
//         sub     eax, edx ; either OF will be set, or the result is already 0x80000000
//         jo      nanExit
//         ret
//
// nanExit:
//         mov     eax, INT32C__INDEFINITE
//         ret
//
// @safeint32OperatorSub@8 endp
//
//================================================================

template <>
struct SafeBinaryImpl<safeint32::Base, OpSub>
{
    struct Code
    {
        static sysinline bool func(safeint32::Base X, safeint32::Base Y, safeint32::Base& R)
        {
            //
            // If the numbers have equal sign, overflow is impossible.
            // If the numbers have different sign, and the result sign is not equal to X sign,
            // we have overflow;
            //

            R = X - Y;

            int32 ovf = (X ^ Y) & (R ^ X); // MSB is overflow flag

            return ovf >= 0;
        }
    };
};

//----------------------------------------------------------------

sysinline safeint32::Base safeint32OperatorSub(safeint32::Base X, safeint32::Base Y)
{
    safeint32::Base R = 0;

    if (X == INT32C__INDEFINITE)
        goto nanExit;

    if (Y == INT32C__INDEFINITE)
        goto nanExit;

    if_not ((SafeBinaryImpl<safeint32::Base, OpSub>::Code::func(X, Y, R)))
        goto nanExit;

    return X - Y;

nanExit:

    return INT32C__INDEFINITE;
}

//================================================================
//
// safeint32OperatorMul
//
// C++ version compiles to ~11 instructions.
//
//----------------------------------------------------------------
//
// @safeint32OperatorMul@8 proc
//
//         cmp     ecx, INT32C__INDEFINITE
//         je      nanExit
//         cmp     edx, INT32C__INDEFINITE
//         je      nanExit
//         mov     eax, ecx
//         imul    eax, edx ; either OF will be set, or the result is already 0x80000000
//         jo      nanExit
//         ret
//
// nanExit:
//         mov     eax, INT32C__INDEFINITE
//         ret
//
// @safeint32OperatorMul@8 endp
//
//================================================================

template <>
struct SafeBinaryImpl<safeint32::Base, OpMul>
{
    struct Code
    {
        //
        // valid signed range is -0x80000000 .. +0x7FFFFFFF;
        //
        // convert to unsigned (to use circular arithmetics mod 2^64):
        // 0xFFFFFFFF80000000 .. 0x000000007FFFFFFF with wrap around zero;
        //
        // add 0x80000000 (rotate circle):
        // 0x0000000000000000 .. 0x00000000FFFFFFFF;
        //
        // on other words, high word of (res64 + 2^31) should be zero
        //

        static sysinline bool func(safeint32::Base X, safeint32::Base Y, safeint32::Base& result)
        {
        #ifdef EMUL_AVAILABLE
            int64 res64 = __emul(X, Y);
        #elif defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)
            int64 res64 = int64(X) * int64(Y);
        #else
            #define Need to implement
        #endif

            uint64 test64 = uint64(res64) + 0x80000000;
            uint32 testHi = test64 >> 32;

            result = int32(res64);
            return (testHi == 0);
        }
    };
};

//----------------------------------------------------------------

sysinline safeint32::Base safeint32OperatorMul(safeint32::Base X, safeint32::Base Y)
{
    safeint32::Base result = 0;

    if (X == INT32C__INDEFINITE)
        goto nanExit;

    if (Y == INT32C__INDEFINITE)
        goto nanExit;

    ////

    if_not ((SafeBinaryImpl<safeint32::Base, OpMul>::Code::func(X, Y, result)))
        goto nanExit;

    ////

    return result;

nanExit:

    return INT32C__INDEFINITE;
}

//================================================================
//
// Basic binary operations: +, -, *
//
//================================================================

#define TMP_MACRO(op, opFunc) \
    \
    sysinline safeint32::Type operator op(const safeint32::Type& X, const safeint32::Type& Y) \
        {return INT32C__CREATE(opFunc(INT32C__ACCESS(X), INT32C__ACCESS(Y)));}

TMP_MACRO(+, safeint32OperatorAdd)
TMP_MACRO(-, safeint32OperatorSub)
TMP_MACRO(*, safeint32OperatorMul)

#undef TMP_MACRO

safeint32::Type operator /(const safeint32::Type& X, const safeint32::Type& Y);
safeint32::Type operator %(const safeint32::Type& X, const safeint32::Type& Y);

//================================================================
//
// "~"
//
//================================================================

sysinline safeint32::Type operator ~(const safeint32::Type& X)
{
    safeint32::Base indefinite = INT32C__INDEFINITE;

    safeint32::Base Xv = INT32C__ACCESS(X);

    safeint32::Base R = ~Xv;

    if (Xv == indefinite)
        R = indefinite;

    return INT32C__CREATE(R);
}

//================================================================
//
// "&", "|", "^"
//
//================================================================

#define TMP_MACRO(OP) \
    \
    sysinline safeint32::Type operator OP(const safeint32::Type& X, const safeint32::Type& Y) \
    { \
        safeint32::Base indefinite = INT32C__INDEFINITE; \
        \
        safeint32::Base Xv = INT32C__ACCESS(X); \
        safeint32::Base Yv = INT32C__ACCESS(Y); \
        \
        safeint32::Base nanArg = (Yv == indefinite) + (Xv == indefinite); \
        \
        safeint32::Base R = Xv OP Yv; \
        \
        if (nanArg) \
            R = indefinite; \
        \
        return INT32C__CREATE(R); \
    }

TMP_MACRO(&)
TMP_MACRO(|)
TMP_MACRO(^)

#undef TMP_MACRO

//================================================================
//
// ++
// --
//
//================================================================

#define TMP_MACRO(incop, baseop) \
    \
    sysinline safeint32::Type& operator incop(safeint32::Type& X) \
    { \
        X = X baseop INT32C__CREATE(1); \
        return X; \
    } \
    \
    sysinline safeint32::Type operator incop(safeint32::Type& X, int) \
    { \
        safeint32::Type tmp = X; \
        X = X baseop INT32C__CREATE(1); \
        return tmp; \
    }

////

TMP_MACRO(++, +)
TMP_MACRO(--, -)

#undef TMP_MACRO

//================================================================
//
// "<<"
//
//================================================================

sysinline safeint32::Type operator <<(const safeint32::Type& X, const safeint32::Type& Y)
{

    safeint32::Base indefinite = INT32C__INDEFINITE;

    safeint32::Base Xv = INT32C__ACCESS(X);
    safeint32::Base Yv = INT32C__ACCESS(Y);

    safeint32::Base nanArg = (Xv == indefinite) | (Yv == indefinite);

    //
    // shift value should be in 0 .. INT32C__BITCOUNT - 1 range
    //

    safeint32::Base badShift = (safeint32::BaseUnsigned(Yv) >= INT32C__BITCOUNT);

    //
    // the result for succesful case
    //

    safeint32::Base R = Xv << Yv;

    //
    // signed numbers should be in 2s complementary code;
    // overflow checking for left shift is performed by shifting the result right
    // and comparing with original value;
    //

    safeint32::Base overflow = (R >> Yv) != Xv;

    if (nanArg | badShift | overflow)
        R = indefinite;

    return INT32C__CREATE(R);

}

//================================================================
//
// ">>"
//
//================================================================

sysinline safeint32::Type operator >>(const safeint32::Type& X, const safeint32::Type& Y)
{

    safeint32::Base indefinite = INT32C__INDEFINITE;

    safeint32::Base Xv = INT32C__ACCESS(X);
    safeint32::Base Yv = INT32C__ACCESS(Y);

    safeint32::Base nanArg = (Xv == indefinite) | (Yv == indefinite);

    //
    // shift value should be in 0 .. INT32C__BITCOUNT - 1 range
    //

    safeint32::Base badShift = (safeint32::BaseUnsigned(Yv) >= INT32C__BITCOUNT);

    //
    // the result for succesful case
    //

    safeint32::Base R = Xv >> Yv;

    if (nanArg | badShift)
        R = indefinite;

    return INT32C__CREATE(R);

}

//================================================================
//
// Binary ops with base type.
//
//================================================================

#define TMP_MACRO(OP) \
    \
    sysinline safeint32::Type operator OP(const safeint32::Type& A, safeint32::Base B) \
        {return A OP INT32C__CREATE(B);} \
    \
    sysinline safeint32::Type operator OP(safeint32::Base A, const safeint32::Type& B) \
        {return INT32C__CREATE(A) OP B;} \

////

TMP_MACRO(+)
TMP_MACRO(-)
TMP_MACRO(*)
TMP_MACRO(/)
TMP_MACRO(%)
TMP_MACRO(<<)
TMP_MACRO(>>)
TMP_MACRO(&)
TMP_MACRO(|)
TMP_MACRO(^)

#undef TMP_MACRO

//================================================================
//
// Binary assignment operations.
//
//================================================================

#define TMP_MACRO(asgop, baseop) \
    \
    sysinline safeint32::Type& operator asgop(safeint32::Type& A, const safeint32::Type& B) \
        {return A = A baseop B;} \
    \
    sysinline safeint32::Type& operator asgop(safeint32::Type& A, safeint32::Base B) \
        {return A = A baseop INT32C__CREATE(B);} \

////

TMP_MACRO(+=, +)
TMP_MACRO(-=, -)
TMP_MACRO(*=, *)
TMP_MACRO(/=, /)
TMP_MACRO(%=, %)
TMP_MACRO(<<=, <<)
TMP_MACRO(>>=, >>)
TMP_MACRO(&=, &)
TMP_MACRO(|=, |)
TMP_MACRO(^=, ^)

#undef TMP_MACRO

//================================================================
//
// comparisons
//
//================================================================

#define TMP_MACRO(OP) \
    \
    sysinline bool operator OP(const safeint32::Type& A, const safeint32::Type& B) \
        {return INT32C__ACCESS(A) OP INT32C__ACCESS(B);} \
    \
    sysinline bool operator OP(const safeint32::Type& A, const safeint32::Base& B) \
        {return INT32C__ACCESS(A) OP (B);} \
    \
    sysinline bool operator OP(const safeint32::Base& A, const safeint32::Type& B) \
        {return (A) OP INT32C__ACCESS(B);}

////

TMP_MACRO(==)
TMP_MACRO(!=)
TMP_MACRO(>)
TMP_MACRO(<)
TMP_MACRO(>=)
TMP_MACRO(<=)

#undef TMP_MACRO

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

template <>
struct DefImpl<safeint32::Type>
{
    static sysinline bool func(const safeint32::Type& V)
    {
        return INT32C__ACCESS(V) != safeint32::indefinite;
    }
};

//================================================================
//
// NanOfImpl
//
//================================================================

template <>
struct NanOfImpl<safeint32::Type>
{
    static sysinline safeint32::Type func()
        {return INT32C__CREATE(safeint32::indefinite);}
};

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
// Base -> Safeint32 (Checked/Unchecked)
//
//================================================================

template <Rounding rounding, ConvertCheck check, ConvertHint hint> // exact and checked
struct ConvertImpl<BuiltinInt, safeint32::Type, check, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, safeint32::Base));
        COMPILE_ASSERT(TYPE_EQUAL(Dst, safeint32::Type));

        static sysinline Dst func(Src src) {return INT32C__CREATE(src);}
    };
};

//================================================================
//
// Base -> Safeint32 (Flag)
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact
struct ConvertImplFlag<BuiltinInt, safeint32::Type, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, safeint32::Base));
        COMPILE_ASSERT(TYPE_EQUAL(Dst, safeint32::Type));

        static sysinline bool func(const Src& src, Dst& dst)
        {
            dst = INT32C__CREATE(src);
            return (src != safeint32::indefinite);
        }
    };
};

//================================================================
//
// Safeint32 -> Base (Unchecked)
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact, unchecked
struct ConvertImpl<safeint32::Type, BuiltinInt, ConvertUnchecked, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, safeint32::Type));
        COMPILE_ASSERT(TYPE_EQUAL(Dst, safeint32::Base));

        static sysinline Dst func(Src src) {return INT32C__ACCESS(src);}
    };
};

//================================================================
//
// Safeint32 -> Base (Flag)
//
//================================================================

template <Rounding rounding, ConvertHint hint> // exact, unchecked
struct ConvertImplFlag<safeint32::Type, BuiltinInt, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, safeint32::Type));
        COMPILE_ASSERT(TYPE_EQUAL(Dst, safeint32::Base));

        static sysinline bool func(const Src& src, Dst& dst)
        {
            dst = INT32C__ACCESS(src);
            return (INT32C__ACCESS(src) != safeint32::indefinite);
        }
    };
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// minv
// maxv
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#define TMP_MACRO(func, baseop) \
    \
    template <> \
    sysinline safeint32::Type func(const safeint32::Type& A, const safeint32::Type& B) \
    { \
        safeint32::Base indefinite = INT32C__INDEFINITE; \
          \
        safeint32::Base Av = INT32C__ACCESS(A); \
        safeint32::Base Bv = INT32C__ACCESS(B); \
        \
        safeint32::Base result = Av baseop Bv ? Av : Bv; \
        \
        bool nan = (Av == indefinite) || (Bv == indefinite); \
        \
        if (nan) \
            result = indefinite; \
        \
        return INT32C__CREATE(result); \
    }

TMP_MACRO(minv, <)
TMP_MACRO(maxv, >)

#undef TMP_MACRO

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// absv
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <>
struct SafeUnaryImpl<safeint32::Base, OpAbs>
{
    struct Code
    {
        static sysinline bool func(safeint32::Base value, safeint32::Base& result)
        {
            result = abs(value);
            return (value != INT32C__INDEFINITE); // Only 0x80000000 cannot be abs-ed
        }
    };
};

//----------------------------------------------------------------

template <>
sysinline safeint32::Type absv(const safeint32::Type& X)
{
    safeint32::Base Xv = INT32C__ACCESS(X);

    safeint32::Base result = abs(Xv);

    safeint32::Base indefinite = INT32C__INDEFINITE;

    COMPILE_ASSERT(safeint32::rangeIsSymmetric);

    if (Xv == indefinite)
        result = indefinite;

    return INT32C__CREATE(result);
}
