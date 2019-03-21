#pragma once

#include "vectorTypes/vectorBase.h"
#include "numbers/interface/numberInterface.h"
#include "vectorTypes/half/halfType.h"
#include "point/point.h"

//================================================================
//
// Vector type registry
//
//================================================================

TYPEOF_REGISTER(0xD386, int8_x2)
TYPEOF_REGISTER(0xBD9C, uint8_x2)
TYPEOF_REGISTER(0x6535, int16_x2)
TYPEOF_REGISTER(0xC671, uint16_x2)
TYPEOF_REGISTER(0x347A, int32_x2)
TYPEOF_REGISTER(0x8BBA, uint32_x2)

TYPEOF_REGISTER(0x4D3F, int8_x4)
TYPEOF_REGISTER(0x8534, uint8_x4)
TYPEOF_REGISTER(0xAD0F, int16_x4)
TYPEOF_REGISTER(0xBDCD, uint16_x4)
TYPEOF_REGISTER(0xF59D, int32_x4)
TYPEOF_REGISTER(0x0F52, uint32_x4)

TYPEOF_REGISTER(0x6CE3, float16_x2)
TYPEOF_REGISTER(0x31AD, float16_x4)

TYPEOF_REGISTER(0xA386, float32_x2)
TYPEOF_REGISTER(0x6F7E, float32_x4)

//================================================================
//
// VectorBaseImpl
//
//================================================================

#define TMP_MACRO(Vector, BaseType) \
    \
    template <> \
    struct VectorBaseImpl<Vector> \
        {using T = BaseType;};

TMP_MACRO(int8_x2, int8)
TMP_MACRO(uint8_x2, uint8)
TMP_MACRO(int16_x2, int16)
TMP_MACRO(uint16_x2, uint16)
TMP_MACRO(int32_x2, int32)
TMP_MACRO(uint32_x2, uint32)
TMP_MACRO(float16_x2, float16)
TMP_MACRO(float32_x2, float32)

TMP_MACRO(int8_x4, int8)
TMP_MACRO(uint8_x4, uint8)
TMP_MACRO(int16_x4, int16)
TMP_MACRO(uint16_x4, uint16)
TMP_MACRO(int32_x4, int32)
TMP_MACRO(uint32_x4, uint32)
TMP_MACRO(float16_x4, float16)
TMP_MACRO(float32_x4, float32)

#undef TMP_MACRO

//================================================================
//
// VectorExtendImpl<>
//
//================================================================

#define TMP_MACRO(Vector) \
    \
    template <> \
    struct VectorExtendImpl<Vector> \
    { \
        static sysinline Vector func(VECTOR_BASE_(Vector) value) \
        { \
            return make_##Vector(value, value); \
        } \
    };

TMP_MACRO(int8_x2)
TMP_MACRO(uint8_x2)
TMP_MACRO(int16_x2)
TMP_MACRO(uint16_x2)
TMP_MACRO(int32_x2)
TMP_MACRO(uint32_x2)
TMP_MACRO(float16_x2)
TMP_MACRO(float32_x2)

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(Vector) \
    \
    template <> \
    struct VectorExtendImpl<Vector> \
    { \
        static sysinline Vector func(VECTOR_BASE_(Vector) value) \
        { \
            return make_##Vector(value, value, value, value); \
        } \
    };

TMP_MACRO(int8_x4)
TMP_MACRO(uint8_x4)
TMP_MACRO(int16_x4)
TMP_MACRO(uint16_x4)
TMP_MACRO(int32_x4)
TMP_MACRO(uint32_x4)
TMP_MACRO(float16_x4)
TMP_MACRO(float32_x4)

#undef TMP_MACRO

//================================================================
//
// MakeVectorType
//
//================================================================

template <typename Base, int rank>
struct MakeVectorType;

#define TMP_MACRO(Base, rank, Result) \
    \
    template <> \
    struct MakeVectorType<Base, rank> \
        {using T = Result;};

TMP_MACRO(int8, 1, int8)
TMP_MACRO(uint8, 1, uint8)
TMP_MACRO(int16, 1, int16)
TMP_MACRO(uint16, 1, uint16)
TMP_MACRO(int32, 1, int32)
TMP_MACRO(uint32, 1, uint32)
TMP_MACRO(float16, 1, float16)
TMP_MACRO(float32, 1, float32)
TMP_MACRO(bool, 1, bool)

TMP_MACRO(int8, 2, int8_x2)
TMP_MACRO(uint8, 2, uint8_x2)
TMP_MACRO(int16, 2, int16_x2)
TMP_MACRO(uint16, 2, uint16_x2)
TMP_MACRO(int32, 2, int32_x2)
TMP_MACRO(uint32, 2, uint32_x2)
TMP_MACRO(float16, 2, float16_x2)
TMP_MACRO(float32, 2, float32_x2)
TMP_MACRO(bool, 2, bool_x2)

TMP_MACRO(int8, 4, int8_x4)
TMP_MACRO(uint8, 4, uint8_x4)
TMP_MACRO(int16, 4, int16_x4)
TMP_MACRO(uint16, 4, uint16_x4)
TMP_MACRO(int32, 4, int32_x4)
TMP_MACRO(uint32, 4, uint32_x4)
TMP_MACRO(float16, 4, float16_x4)
TMP_MACRO(float32, 4, float32_x4)
TMP_MACRO(bool, 4, bool_x4)

#undef TMP_MACRO

//================================================================
//
// MakeVectorType
//
//================================================================

template <typename Base, int rank>
struct MakeVectorType<const Base, rank>
{
    using T = const typename MakeVectorType<Base, rank>::T;
};

//================================================================
//
// VectorRebaseImpl
//
//================================================================

#define TMP_MACRO(Vector, rank) \
    \
    template <typename NewBase> \
    struct VectorRebaseImpl<Vector, NewBase> \
        {using T = typename MakeVectorType<NewBase, rank>::T;};

TMP_MACRO(int8_x2, 2)
TMP_MACRO(uint8_x2, 2)
TMP_MACRO(int16_x2, 2)
TMP_MACRO(uint16_x2, 2)
TMP_MACRO(int32_x2, 2)
TMP_MACRO(uint32_x2, 2)
TMP_MACRO(float16_x2, 2)
TMP_MACRO(float32_x2, 2)

TMP_MACRO(int8_x4, 4)
TMP_MACRO(uint8_x4, 4)
TMP_MACRO(int16_x4, 4)
TMP_MACRO(uint16_x4, 4)
TMP_MACRO(int32_x4, 4)
TMP_MACRO(uint32_x4, 4)
TMP_MACRO(float16_x4, 4)
TMP_MACRO(float32_x4, 4)

#undef TMP_MACRO

//================================================================
//
// VectorTypeRank
//
//================================================================

template <typename Type>
struct VectorTypeRank
{
    static const int val = sizeof(Type) / sizeof(VECTOR_BASE(Type));
};

//================================================================
//
// makeVec2
// makeVec4
//
// Generalized functions
//
//================================================================

template <typename Base>
sysinline typename MakeVectorType<Base, 2>::T makeVec2(const Base& x, const Base& y)
{
    typename MakeVectorType<Base, 2>::T tmp;
    tmp.x = x;
    tmp.y = y;
    return tmp;
}

//----------------------------------------------------------------

template <typename Base>
sysinline typename MakeVectorType<Base, 4>::T makeVec4(const Base& x, const Base& y, const Base& z, const Base& w)
{
    typename MakeVectorType<Base, 4>::T tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    return tmp;
}

//================================================================
//
// VectorX2
//
//================================================================

struct VectorX2;

CONVERT_FAMILY_IMPL(int8_x2, VectorX2)
CONVERT_FAMILY_IMPL(uint8_x2, VectorX2)
CONVERT_FAMILY_IMPL(int16_x2, VectorX2)
CONVERT_FAMILY_IMPL(uint16_x2, VectorX2)
CONVERT_FAMILY_IMPL(int32_x2, VectorX2)
CONVERT_FAMILY_IMPL(uint32_x2, VectorX2)
CONVERT_FAMILY_IMPL(float16_x2, VectorX2)
CONVERT_FAMILY_IMPL(float32_x2, VectorX2)

//================================================================
//
// VectorX4
//
//================================================================

struct VectorX4;

CONVERT_FAMILY_IMPL(int8_x4, VectorX4)
CONVERT_FAMILY_IMPL(uint8_x4, VectorX4)
CONVERT_FAMILY_IMPL(int16_x4, VectorX4)
CONVERT_FAMILY_IMPL(uint16_x4, VectorX4)
CONVERT_FAMILY_IMPL(int32_x4, VectorX4)
CONVERT_FAMILY_IMPL(uint32_x4, VectorX4)
CONVERT_FAMILY_IMPL(float16_x4, VectorX4)
CONVERT_FAMILY_IMPL(float32_x4, VectorX4)

//================================================================
//
// VectorX2 -> VectorX2
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<VectorX2, VectorX2, check, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(Src);
        using DstBase = VECTOR_BASE(Dst);

        using BaseConvert = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Dst func(const Src& src)
        {
            return makeVec2<DstBase>
            (
                BaseConvert::func(src.x),
                BaseConvert::func(src.y)
            );
        }
    };
};

//================================================================
//
// VectorX4 -> VectorX4
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<VectorX4, VectorX4, check, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(Src);
        using DstBase = VECTOR_BASE(Dst);

        using BaseConvert = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Dst func(const Src& src)
        {
            return makeVec4<DstBase>
            (
                BaseConvert::func(src.x),
                BaseConvert::func(src.y),
                BaseConvert::func(src.z),
                BaseConvert::func(src.w)
            );
        }
    };
};

//================================================================
//
// VectorX2 -> Point
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<VectorX2, PointFamily, check, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(Src);
        using DstBase = VECTOR_BASE(Dst);

        using BaseConvert = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Dst func(const Src& src)
        {
            return point<DstBase>
            (
                BaseConvert::func(src.x),
                BaseConvert::func(src.y)
            );
        }
    };
};

//================================================================
//
// Point -> VectorX2
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<PointFamily, VectorX2, check, rounding, hint>
{
    template <typename Src, typename Dst>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(Src);
        using DstBase = VECTOR_BASE(Dst);

        using BaseConvert = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline Dst func(const Src& src)
        {
            return makeVec2<DstBase>
            (
                BaseConvert::func(src.X),
                BaseConvert::func(src.Y)
            );
        }
    };
};

//================================================================
//
// exchange
//
//================================================================

VECTOR_INT_X2_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)
VECTOR_INT_X4_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)
VECTOR_FLOAT_X2_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)
VECTOR_FLOAT_X4_FOREACH(EXCHANGE_DEFINE_SIMPLE, _)
