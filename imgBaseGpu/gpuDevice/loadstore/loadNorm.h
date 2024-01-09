#pragma once

#include "gpuDevice/loadViaSamplerCache.h"
#include "imageRead/loadMode.h"
#include "vectorTypes/vectorType.h"
#include "dbgptr/dbgptrProtos.h"

//================================================================
//
// ImportNormalizedValue
//
//================================================================

template <typename Src>
struct ImportNormalizedValue;

////

template <typename Src>
sysinline auto importNormalizedValue(const Src& src)
    {return ImportNormalizedValue<Src>::func(src);}

//================================================================
//
// ImportNormalizedValue<FLOAT>
//
//================================================================

template <>
struct ImportNormalizedValue<float16>
{
    static sysinline float32 func(const float16& value)
        {return convertFloat32(value);}
};

template <>
struct ImportNormalizedValue<float32>
{
    static sysinline float32 func(const float32& value)
        {return value;}
};

//================================================================
//
// ImportNormalizedValue<INT>
//
//================================================================

template <>
struct ImportNormalizedValue<int8>
{
    static sysinline float32 func(const int8& value)
        {return value * (1.f / 0x7F);}
};

template <>
struct ImportNormalizedValue<int16>
{
    static sysinline float32 func(const int16& value)
        {return value * (1.f / 0x7FFF);}
};

//----------------------------------------------------------------

template <>
struct ImportNormalizedValue<uint8>
{
    static sysinline float32 func(const uint8& value)
        {return value * (1.f / 0xFF);}
};

template <>
struct ImportNormalizedValue<uint16>
{
    static sysinline float32 func(const uint16& value)
        {return value * (1.f / 0xFFFF);}
};

//================================================================
//
// ImportNormalizedValue<VEC2>
//
//================================================================

#define TMP_MACRO(Type) \
    \
    template <> \
    struct ImportNormalizedValue<Type> \
    { \
        static sysinline auto func(const Type& value) \
        { \
            return makeVec2 \
            ( \
                ImportNormalizedValue<decltype(value.x)>::func(value.x), \
                ImportNormalizedValue<decltype(value.y)>::func(value.y) \
            ); \
        } \
    };

TMP_MACRO(int8_x2)
TMP_MACRO(int16_x2)

TMP_MACRO(uint8_x2)
TMP_MACRO(uint16_x2)

TMP_MACRO(float16_x2)
TMP_MACRO(float32_x2)

#undef TMP_MACRO

//================================================================
//
// ImportNormalizedValue<VEC4>
//
//================================================================

#define TMP_MACRO(Type) \
    \
    template <> \
    struct ImportNormalizedValue<Type> \
    { \
        static sysinline auto func(const Type& value) \
        { \
            return makeVec4 \
            ( \
                ImportNormalizedValue<decltype(value.x)>::func(value.x), \
                ImportNormalizedValue<decltype(value.y)>::func(value.y), \
                ImportNormalizedValue<decltype(value.z)>::func(value.z), \
                ImportNormalizedValue<decltype(value.w)>::func(value.w) \
            ); \
        } \
    };

TMP_MACRO(int8_x4)
TMP_MACRO(int16_x4)

TMP_MACRO(uint8_x4)
TMP_MACRO(uint16_x4)

TMP_MACRO(float16_x4)
TMP_MACRO(float32_x4)

#undef TMP_MACRO

//================================================================
//
// loadNorm
//
//================================================================

template <typename Pointer>
sysinline auto loadNorm(Pointer srcPtr)
{
    return importNormalizedValue(helpRead(*srcPtr));
}

template <typename Pointer>
sysinline auto loadNormViaSamplerCache(Pointer srcPtr)
{
    return importNormalizedValue(loadViaSamplerCache(&helpRead(*srcPtr)));
}
