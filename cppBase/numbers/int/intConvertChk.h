#pragma once

#include "numbers/interface/numberInterface.h"

namespace intConvertChk {

//================================================================
//
// Optimized int-int conversion test
//
//================================================================

//================================================================
//
// FitTemplate
//
//================================================================

template <bool srcSigned, bool dstSigned>
struct FitTemplate;

//================================================================
//
// Signed -> Signed
//
//================================================================

template <>
struct FitTemplate<true, true>
{
    template <typename Src, typename Dst>
    struct Code
    {
        static sysinline bool func(Src src)
        {
            return
                (
                    src >= TYPE_MIN(Dst) ||
                    TYPE_MIN(Src) >= TYPE_MIN(Dst)
                )
                &&
                (
                    src <= TYPE_MAX(Dst) ||
                    TYPE_MAX(Src) <= TYPE_MAX(Dst)
                );
        }
    };
};

//================================================================
//
// Unsigned -> Unsigned
//
//================================================================

template <>
struct FitTemplate<false, false>
{
    template <typename Src, typename Dst>
    struct Code
    {
        static sysinline bool func(Src src)
        {
            return
                src <= TYPE_MAX(Dst) ||
                TYPE_MAX(Src) <= TYPE_MAX(Dst);
        }
    };
};

//================================================================
//
// Signed -> Unsigned
//
//================================================================

template <>
struct FitTemplate<true, false>
{
    template <typename Src, typename Dst>
    struct Code
    {
        static sysinline bool func(Src value)
        {
            using SrcU = TYPE_MAKE_UNSIGNED(Src);

            return
                (
                    value >= 0
                )
                &&
                (
                    SrcU(value) <= TYPE_MAX(Dst) ||
                    SrcU(TYPE_MAX(Src)) <= TYPE_MAX(Dst)
                );
        }
    };
};

//================================================================
//
// Unsigned -> Signed
//
//================================================================

template <>
struct FitTemplate<false, true>
{
    template <typename Src, typename Dst>
    struct Code
    {
        static sysinline bool func(Src src)
        {
            using DstU = TYPE_MAKE_UNSIGNED(Dst);

            return
                src <= DstU(TYPE_MAX(Dst)) ||
                TYPE_MAX(Src) <= DstU(TYPE_MAX(Dst));
        }
    };
};

//================================================================
//
// Fit
//
//================================================================

template <typename Src, typename Dst>
struct Fit
{
    using Code = typename FitTemplate<TYPE_IS_SIGNED(Src), TYPE_IS_SIGNED(Dst)>::template Code<Src, Dst>;
};

//----------------------------------------------------------------

template <typename Type>
struct Fit<Type, Type> // Optimization
{
    struct Code
    {
        static sysinline bool func(Type value)
            {return true;}
    };
};

//----------------------------------------------------------------

}
