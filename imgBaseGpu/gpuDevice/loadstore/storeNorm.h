#pragma once

#include "compileTools/compileTools.h"
#include "numbers/float/floatType.h"
#include "numbers/float16/float16Type.h"
#include "vectorTypes/vectorBase.h"
#include "dbgptr/dbgptrGate.h"
#include "data/pointerInterface.h"

//================================================================
//
// float_XX_rndsat
//
// Converts float value to the specified integer type
// with rounding and clamping to the destination range.
//
//================================================================

#if defined(__CUDA_ARCH__)

//----------------------------------------------------------------

__device__ uint32 float_u8_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.u8.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

__device__ uint32 float_u16_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.u16.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

__device__ uint32 float_u32_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.u32.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

//----------------------------------------------------------------

__device__ uint32 float_s8_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.s8.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

__device__ uint32 float_s16_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.s16.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

__device__ uint32 float_s32_rndsat(float32 value)
    {uint32 result; asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(value)); return result;}

//----------------------------------------------------------------

#endif

//================================================================
//
// convertRoundSaturate
//
// Converts a float value to a specified integer type,
// with rounding and clamping to the integer range.
//
//================================================================

template <typename Dst, typename Src>
sysinline Dst convertRoundSaturate(const Src& src);

//
// integer 1-component
//

#define TMP_MACRO(Src, Dst, expression) \
    \
    template <> \
    sysinline Dst convertRoundSaturate(const Src& src) \
        {return expression;}


#if defined(__CUDA_ARCH__)

    TMP_MACRO(float32, int8,  float_s8_rndsat(src))
    TMP_MACRO(float32, int16, float_s16_rndsat(src))
    TMP_MACRO(float32, int32, float_s32_rndsat(src))

    TMP_MACRO(float32, uint8,  float_u8_rndsat(src))
    TMP_MACRO(float32, uint16, float_u16_rndsat(src))
    TMP_MACRO(float32, uint32, float_u32_rndsat(src))

#else

    TMP_MACRO(float32, int8,  convertNearest<int32>(clampRange<float32>(src, -0x7F, +0x7F)))
    TMP_MACRO(float32, int16, convertNearest<int32>(clampRange<float32>(src, -0x7FFF, +0x7FFF)))

    TMP_MACRO(float32, uint8,  convertNearest<int32>(clampRange<float32>(src, 0, 0xFF)))
    TMP_MACRO(float32, uint16, convertNearest<int32>(clampRange<float32>(src, 0, 0xFFFF)))

    //
    // 32-bit numbers need special approach:
    // Compare only to exactly-representable floats (!)
    //

    template <>
    sysinline int32 convertRoundSaturate(const float32& src)
    {
        auto result = convertNearest<int32>(src);

        if (src <= -2147483648.f)
            result = -2147483648;

        if (src >= +2147483648.f)
            result = 2147483647;

        return result;
    }

    template <>
    sysinline uint32 convertRoundSaturate(const float32& src)
    {
        auto result = convertNearest<uint32>(src);

        if (src < 0)
            result = 0;

        if (src >= 4294967296)
            result = 4294967295;

        return result;
    }

#endif

#undef TMP_MACRO

//
// integer 2-component
//

#define TMP_MACRO(Src, Dst, DstBase) \
    \
    template <> \
    sysinline Dst convertRoundSaturate(const Src& src) \
    { \
        return make_##Dst \
        ( \
            convertRoundSaturate<DstBase>(src.x), \
            convertRoundSaturate<DstBase>(src.y) \
        ); \
    }

TMP_MACRO(float32_x2, uint8_x2, uint8)
TMP_MACRO(float32_x2, uint16_x2, uint16)
TMP_MACRO(float32_x2, uint32_x2, uint32)

TMP_MACRO(float32_x2, int8_x2, int8)
TMP_MACRO(float32_x2, int16_x2, int16)
TMP_MACRO(float32_x2, int32_x2, int32)

#undef TMP_MACRO

//
// integer 4-component
//

#define TMP_MACRO(Src, Dst, DstBase) \
    \
    template <> \
    sysinline Dst convertRoundSaturate(const Src& src) \
    { \
        return make_##Dst \
        ( \
            convertRoundSaturate<DstBase>(src.x), \
            convertRoundSaturate<DstBase>(src.y), \
            convertRoundSaturate<DstBase>(src.z), \
            convertRoundSaturate<DstBase>(src.w) \
        ); \
    }

TMP_MACRO(float32_x4, uint8_x4, uint8)
TMP_MACRO(float32_x4, uint16_x4, uint16)
TMP_MACRO(float32_x4, uint32_x4, uint32)

TMP_MACRO(float32_x4, int8_x4, int8)
TMP_MACRO(float32_x4, int16_x4, int16)
TMP_MACRO(float32_x4, int32_x4, int32)

#undef TMP_MACRO

//================================================================
//
// convertNormClamp
//
// Converts normalized float to memory integer type,
// the src range is [0, 1] for unsigned types and [-1, +1] for signed types.
//
//================================================================

template <typename Dst, typename Src>
sysinline Dst convertNormClamp(const Src& src)
    MISSING_FUNCTION_BODY

//
// float32 (copy)
//

#define TMP_MACRO(Type) \
    template <> \
    sysinline Type convertNormClamp(const Type& src) \
        {return src;}

TMP_MACRO(float32)
TMP_MACRO(float32_x2)
TMP_MACRO(float32_x4)

#undef TMP_MACRO

//
// float16 (pack)
//

template <>
sysinline float16 convertNormClamp(const float32& src) \
    {return packFloat16(src);}

//
// integer 1-component
//

#define TMP_MACRO(Src, Dst, expression) \
    \
    template <> \
    sysinline Dst convertNormClamp(const Src& src) \
        {return expression;}

//
//
//

#if defined(__CUDA_ARCH__)

    TMP_MACRO(float32, int8,  float_s8_rndsat(src * 0x7F))
    TMP_MACRO(float32, int16, float_s16_rndsat(src * 0x7FFF))

    TMP_MACRO(float32, uint8,  float_u8_rndsat(src * 0xFF))
    TMP_MACRO(float32, uint16, float_u16_rndsat(src * 0xFFFF))

#else

    TMP_MACRO(float32, int8,  convertRoundSaturate<int8>(src * 0x7F))
    TMP_MACRO(float32, int16, convertRoundSaturate<int8>(src * 0x7FFF))

    TMP_MACRO(float32, uint8,  convertRoundSaturate<uint8>(src * 0xFF))
    TMP_MACRO(float32, uint16, convertRoundSaturate<uint8>(src * 0xFFFF))

#endif

#undef TMP_MACRO

//
// Vector 2-component
//

#define TMP_MACRO(Src, Dst, DstBase) \
    \
    template <> \
    sysinline Dst convertNormClamp(const Src& src) \
    { \
        return make_##Dst \
        ( \
            convertNormClamp<DstBase>(src.x), \
            convertNormClamp<DstBase>(src.y) \
        ); \
    }

TMP_MACRO(float32_x2, uint8_x2, uint8)
TMP_MACRO(float32_x2, int8_x2, int8)

TMP_MACRO(float32_x2, uint16_x2, uint16)
TMP_MACRO(float32_x2, int16_x2, int16)

TMP_MACRO(float32_x2, float16_x2, float16)

#undef TMP_MACRO

//
// integer 4-component
//

#define TMP_MACRO(Src, Dst, DstBase) \
    \
    template <> \
    sysinline Dst convertNormClamp(const Src& src) \
    { \
        return make_##Dst \
        ( \
            convertNormClamp<DstBase>(src.x), \
            convertNormClamp<DstBase>(src.y), \
            convertNormClamp<DstBase>(src.z), \
            convertNormClamp<DstBase>(src.w) \
        ); \
    }

TMP_MACRO(float32_x4, uint8_x4, uint8)
TMP_MACRO(float32_x4, int8_x4, int8)

TMP_MACRO(float32_x4, uint16_x4, uint16)
TMP_MACRO(float32_x4, int16_x4, int16)

TMP_MACRO(float32_x4, float16_x4, float16)

#undef TMP_MACRO

//================================================================
//
// storeNorm
//
//================================================================

template <typename SrcValue, typename DstPointer>
sysinline void storeNorm(DstPointer dstPtr, const SrcValue& srcValue)
{
    using Dst = typename PtrElemType<DstPointer>::T;
    *dstPtr = convertNormClamp<Dst>(srcValue);
}
