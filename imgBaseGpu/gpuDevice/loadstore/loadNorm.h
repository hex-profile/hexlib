#pragma once

#include "vectorTypes/vectorType.h"
#include "data/gpuPtr.h"
#include "imageRead/loadMode.h"
#include "gpuDevice/loadViaSamplerCache.h"

//================================================================
//
// loadNormCore<float32>
//
//================================================================

template <typename LoadElement>
sysinline float32 loadNormCore(const float32* src)
    {return LoadElement::func(src);}

template <typename LoadElement>
sysinline float32_x2 loadNormCore(const float32_x2* src)
    {return LoadElement::func(src);}

template <typename LoadElement>
sysinline float32_x4 loadNormCore(const float32_x4* src)
    {return LoadElement::func(src);}

//================================================================
//
// loadNormCore<float16>
//
//================================================================

template <typename LoadElement>
sysinline float32 loadNormCore(const float16* src)
    {return convertFloat32(LoadElement::func(src));}

template <typename LoadElement>
sysinline float32_x2 loadNormCore(const float16_x2* src)
    {return convertFloat32(LoadElement::func(src));}

template <typename LoadElement>
sysinline float32_x4 loadNormCore(const float16_x4* src)
    {return convertFloat32(LoadElement::func(src));}

//================================================================
//
// loadNormCore<uint8>
//
//================================================================

template <typename LoadElement>
sysinline float32 loadNormCore(const uint8* src)
{
    return (LoadElement::func(src)) * (1.f / 0xFF);
}

template <typename LoadElement>
sysinline float32_x2 loadNormCore(const uint8_x2* src)
{
    uint8_x2 value = LoadElement::func(src);

    return make_float32_x2
    (
        value.x * (1.f / 0xFF),
        value.y * (1.f / 0xFF)
    );
}

template <typename LoadElement>
sysinline float32_x4 loadNormCore(const uint8_x4* src)
{
    uint8_x4 value = LoadElement::func(src);

    return make_float32_x4
    (
        value.x * (1.f / 0xFF),
        value.y * (1.f / 0xFF),
        value.z * (1.f / 0xFF),
        value.w * (1.f / 0xFF)
    );
}

//================================================================
//
// loadNormCore<int8>
//
//================================================================

template <typename LoadElement>
sysinline float32 loadNormCore(const int8* src)
{
    return (LoadElement::func(src)) * (1.f / 0x7F);
}

template <typename LoadElement>
sysinline float32_x2 loadNormCore(const int8_x2* src)
{
    int8_x2 value = LoadElement::func(src);

    return make_float32_x2
    (
        value.x * (1.f / 0x7F),
        value.y * (1.f / 0x7F)
    );
}

template <typename LoadElement>
sysinline float32_x4 loadNormCore(const int8_x4* src)
{
    int8_x4 value = LoadElement::func(src);

    return make_float32_x4
    (
        value.x * (1.f / 0x7F),
        value.y * (1.f / 0x7F),
        value.z * (1.f / 0x7F),
        value.w * (1.f / 0x7F)
    );
}

//================================================================
//
// loadNormCore<int16>
//
//================================================================

template <typename LoadElement>
sysinline float32 loadNormCore(const int16* src)
{
    return (LoadElement::func(src)) * (1.f / 0x7FFF);
}

template <typename LoadElement>
sysinline float32_x2 loadNormCore(const int16_x2* src)
{
    int16_x2 value = LoadElement::func(src);

    return make_float32_x2
    (
        value.x * (1.f / 0x7FFF),
        value.y * (1.f / 0x7FFF)
    );
}

template <typename LoadElement>
sysinline float32_x4 loadNormCore(const int16_x4* src)
{
    int16_x4 value = LoadElement::func(src);

    return make_float32_x4
    (
        value.x * (1.f / 0x7FFF),
        value.y * (1.f / 0x7FFF),
        value.z * (1.f / 0x7FFF),
        value.w * (1.f / 0x7FFF)
    );
}

//================================================================
//
// loadNorm
//
//================================================================

template <typename Pointer>
sysinline auto loadNorm(Pointer srcPtr)
{
    return loadNormCore<LoadNormal>(unsafePtr(srcPtr, 1));
}

template <typename Pointer>
sysinline auto loadNormViaSamplerCache(Pointer srcPtr)
{
    return loadNormCore<LoadViaSamplerCache>(unsafePtr(srcPtr, 1));
}
