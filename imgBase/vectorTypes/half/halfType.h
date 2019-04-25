#pragma once

#include "numbers/float/floatType.h"
#include "vectorTypes/half/halfBase.h"
#include "vectorTypes/half/halfConvert.h"

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

#define TMP_MACRO(Type, o) \
    template <> struct TypeIsBuiltinFloat<Type> {static const bool result = true;};

TMP_MACRO(float16, o)

#undef TMP_MACRO

//================================================================
//
// TypeIsSignedImpl
//
//================================================================

TYPE_IS_SIGNED_IMPL(float16, true)

//================================================================
//
// TypeMakeSignedImpl
// TypeMakeUnsignedImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    TYPE_MAKE_SIGNED_IMPL(Type, Type) \
    TYPE_MAKE_UNSIGNED_IMPL(Type, Type)

TMP_MACRO(float16, o)

#undef TMP_MACRO

//================================================================
//
// TypeIsControlledImpl
//
//================================================================

TYPE_IS_CONTROLLED_IMPL(float16, true)

//================================================================
//
// TypeMakeControlledImpl
// TypeMakeUncontrolledImpl
//
//================================================================

#define TMP_MACRO(Type, o) \
    TYPE_MAKE_CONTROLLED_IMPL(Type, Type) \
    TYPE_MAKE_UNCONTROLLED_IMPL(Type, Type)

TMP_MACRO(float16, o)

#undef TMP_MACRO

//================================================================
//
// TypeMinMaxImpl
//
//================================================================

template <>
struct TypeMinMaxImpl<float16>
{
    static sysinline float16 minVal() {return packFloat16(-65504.f);}
    static sysinline float16 maxVal() {return packFloat16(+65504.f);}
};

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
// nanOfImpl
//
//================================================================

template <>
sysinline float16 nanOfImpl<float16>()
{
    return packFloat16(nanOfImpl<float32>());
}

//================================================================
//
// def
//
//================================================================

template <>
struct DefImpl<float16>
{
    static sysinline bool func(const float16& value)
    {
        return def(unpackFloat16(value));
    }
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
// ConvertFamilyImpl<>
//
//================================================================

struct BuiltinFloat16;

CONVERT_FAMILY_IMPL(float16, BuiltinFloat16)

//================================================================
//
// float16 -> AnyType
//
// Always goes through float32.
//
//================================================================

template <typename DstFamily, ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<BuiltinFloat16, DstFamily, check, rounding, hint>
{
    using BackendConvert = ConvertImpl<BuiltinFloat, DstFamily, check, rounding, hint>;

    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& src)
        {
            float32 srcFloat = unpackFloat16(src); // no round errors
            return BackendConvert::template Convert<float32, Dst>::func(srcFloat);
        }
    };
};

//----------------------------------------------------------------

template <typename DstFamily, Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<BuiltinFloat16, DstFamily, rounding, hint>
{
    using BackendConvert = ConvertImplFlag<BuiltinFloat, DstFamily, rounding, hint>;

    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& src, Dst& dst)
        {
            float32 srcFloat = unpackFloat16(src); // no round errors
            return BackendConvert::template Convert<float32, Dst>::func(srcFloat, dst);
        }
    };
};

//================================================================
//
// AnyType -> float16
//
// Always goes through float32.
// Only nearest rounding is supported.
//
//================================================================

template <typename SrcFamily, ConvertCheck check, ConvertHint hint>
struct ConvertImpl<SrcFamily, BuiltinFloat16, check, RoundNearest, hint>
{
    using FrontendConvert = ConvertImpl<SrcFamily, BuiltinFloat, check, RoundNearest, hint>;

    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline Dst func(const Src& src)
        {
            float32 srcFloat = FrontendConvert::template Convert<Src, float32>::func(src);
            return packFloat16(srcFloat); // Automatic error checking
        }
    };
};

//----------------------------------------------------------------

template <typename SrcFamily, ConvertHint hint>
struct ConvertImplFlag<SrcFamily, BuiltinFloat16, RoundNearest, hint>
{
    using FrontendConvert = ConvertImpl<SrcFamily, BuiltinFloat, ConvertChecked, RoundNearest, hint>;

    template <typename Src, typename Dst>
    struct Convert
    {
        static sysinline bool func(const Src& src, Dst& dst)
        {
            float32 srcFloat = FrontendConvert::template Convert<Src, float32>::func(src);
            dst = packFloat16(srcFloat);
            return def(dst); // Automatic error checking
        }
    };
};

//================================================================
//
// float16 -> float16
//
//================================================================

template <ConvertCheck check, ConvertHint hint, Rounding rounding>
struct ConvertImpl<BuiltinFloat16, BuiltinFloat16, check, rounding, hint>
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

template <ConvertHint hint, Rounding rounding>
struct ConvertImplFlag<BuiltinFloat16, BuiltinFloat16, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        COMPILE_ASSERT(TYPE_EQUAL(Src, Dst));

        static sysinline bool func(const Src& src, Dst& dst)
            {dst = src; return true;}
    };
};
