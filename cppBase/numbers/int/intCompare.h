#pragma once

#include "numbers/interface/numberInterface.h"

namespace intCompare {

//================================================================
//
// Correct comparison of built-in integer types, including signed to unsigned.
//
//----------------------------------------------------------------
//
// Problem formulation.
//
// In C/C++ we can't safely compare signed to unsigned integers:
//
// int A = -1;
// unsigned int B = 3;
//
// if (A < B) // Never happens!
//   ;
//
//================================================================

//================================================================
//
// Less comparison
//
//================================================================

template <bool signedA, bool signedB>
struct LessTemplate;

//----------------------------------------------------------------

template <>
struct LessTemplate<true, true>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // Both are signed
            return (A < B);
        }
    };
};

//----------------------------------------------------------------

template <>
struct LessTemplate<false, false>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // Both are unsigned
            return (A < B);
        }
    };
};

//----------------------------------------------------------------

template <>
struct LessTemplate<true, false>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // If the signed argument is negative, result is true;
            // otherwise we can compare in unsigned types
            TYPE_MAKE_UNSIGNED(TA) UA = A;
            TYPE_MAKE_UNSIGNED(TB) UB = B;

            return (A < 0) || (UA < UB);
        }
    };
};

//----------------------------------------------------------------

template <>
struct LessTemplate<false, true>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // The signed argument should be non-negative (otherwise result is false);
            // after checking this we can compare in unsigned types
            TYPE_MAKE_UNSIGNED(TA) UA = A;
            TYPE_MAKE_UNSIGNED(TB) UB = B;

            return (B >= 0) && (UA < UB);
        }
    };
};

//================================================================
//
// Equal comparison
//
//================================================================

template <bool signedA, bool signedB>
struct EqualTemplate;

//----------------------------------------------------------------

template <>
struct EqualTemplate<true, true>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // both are signed

            return (A == B);
        }
    };
};

//----------------------------------------------------------------

template <>
struct EqualTemplate<false, false>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // both are unsigned

            return (A == B);
        }
    };
};

//----------------------------------------------------------------

template <>
struct EqualTemplate<true, false>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // comparing signed and unsigned:
            // to be equal, the signed value should be non-negative

            TYPE_MAKE_UNSIGNED(TA) UA = A;
            TYPE_MAKE_UNSIGNED(TB) UB = B;

            return (A >= 0) && (UA == UB);
        }
    };
};

//----------------------------------------------------------------

template <>
struct EqualTemplate<false, true>
{
    template <typename TA, typename TB>
    struct Code
    {
        static inline bool _(TA A, TB B)
        {
            // comparing signed and unsigned:
            // to be equal, the signed value should be non-negative

            TYPE_MAKE_UNSIGNED(TA) UA = A;
            TYPE_MAKE_UNSIGNED(TB) UB = B;

            return (B >= 0) && (UA == UB);
        }
    };
};

//================================================================
//
// Less
//
//================================================================

template <typename TA, typename TB>
struct Less
{
    using Code = typename LessTemplate<TYPE_IS_SIGNED(TA), TYPE_IS_SIGNED(TB)>::template Code<TA, TB>;
};

//================================================================
//
// Equal
//
//================================================================

template <typename TA, typename TB>
struct Equal
{
    using Code = typename EqualTemplate<TYPE_IS_SIGNED(TA), TYPE_IS_SIGNED(TB)>::template Code<TA, TB>;
};

//================================================================
//
// Service functions
//
//================================================================

template <typename TA, typename TB>
inline bool intLess(TA A, TB B)
{
    return Less<TA, TB>::Code::_(A, B);
}

//----------------------------------------------------------------

template <typename TA, typename TB>
inline bool intGreater(TA A, TB B)
{
    return Less<TB, TA>::Code::_(B, A);
}

//----------------------------------------------------------------

template <typename TA, typename TB>
inline bool intEqual(TA A, TB B)
{
    return Equal<TA, TB>::Code::_(A, B);
}

//----------------------------------------------------------------

template <typename TA, typename TB>
inline bool intNotEqual(TA A, TB B)
{
    return !Equal<TA, TB>::Code::_(A, B);
}

//----------------------------------------------------------------

template <typename TA, typename TB>
inline bool intLessEq(TA A, TB B)
{
    return !Less<TB, TA>::Code::_(B, A);
}

//----------------------------------------------------------------

template <typename TA, typename TB>
inline bool intGreaterEq(TA A, TB B)
{
    return !Less<TA, TB>::Code::_(A, B);
}

//----------------------------------------------------------------

template <typename T1, typename T2, typename T3>
inline bool intInRange(T1 X, T2 minVal, T3 maxVal)
{
    return greaterEq(X, minVal) && lessEq(X, maxVal);
}

//----------------------------------------------------------------

}

using intCompare::intLess;
using intCompare::intGreater;
using intCompare::intEqual;
using intCompare::intNotEqual;
using intCompare::intLessEq;
using intCompare::intGreaterEq;
using intCompare::intInRange;
