#include "halfConvertEmu.h"

#include <cmath>

#include "numbers/int/intType.h"

//================================================================
//
// shrRoundLimited
//
// Works correctly for n in [0, 31].
//
// Performs rounding. If the remainder is exactly 1/2,
// it rounds up only for even integer part.
//
//================================================================

inline uint32 shrRoundLimited(uint32 value, int32 n)
{
    uint32 npower = int32(1) << n; // works for n = 0..31

    uint32 mask = npower - 1;
    uint32 half = npower >> 1;

    uint32 remainder = value & mask;

    uint32 result = value >> n;

    if (remainder > half || ((remainder == half) && (result & 1)))
        result += 1;

    return result;
}

//================================================================
//
// packFloat16
//
//================================================================

float16 packFloat16(const float32& value)
{
    uint32 srcUint = * (const uint32*) &value;

    //
    // Sign
    //

    uint32 sign = srcUint >> 31;
    srcUint &= 0x7FFFFFFF;

    ////

    int32 srcBiasedExponent = srcUint >> 23;
    int32 srcOriginalMantissa = srcUint & 0x7FFFFF; // 23 bits

    bool srcNan = srcBiasedExponent == 0xFF && srcOriginalMantissa != 0;

    //
    // Zero and denormals
    //
    // For float16, all float32 zero and subnormals will be zero,
    // regardless of original mantissa. No special handling required.
    //

    uint32 srcMantissa = srcOriginalMantissa | 0x800000;

    //
    //
    //

    int32 exponent = srcBiasedExponent - 0x7F;

    int32 dstExponent = exponent;
    uint32 dstMantissa = 0;

    //
    // Handle potential subnormals
    //

    if (exponent < -14)
    {
        int32 shift = -1 - exponent;

        dstMantissa = shrRoundLimited(srcMantissa, shift); // shr >= +14, so the range is [0, 2^10]

        if (shift >= 25)
            dstMantissa = 0;

        dstExponent = -15;

        if (dstMantissa == 0x400) // out of subnormals
            dstExponent = -14; // mantissa will be zero after & 0x3FF
    }
    else
    {
        dstMantissa = shrRoundLimited(srcMantissa, 13); // .23 -> .10, range [0, 2^11]

        if (dstMantissa == 0x800) // out of mantissa
            ++dstExponent; // mantissa will be zero after & 0x3FF
    }

    //
    // Handle result infinity
    //

    if (dstExponent >= 16)
    {
        dstExponent = 16;
        dstMantissa = 0;
    }

    //
    // Make the result
    //

    uint32 result = (sign << 15) | ((dstExponent + 15) << 10) | (dstMantissa & 0x3FF);

    //
    // Handle src NAN
    //

    if (srcNan)
        result = 0x7FFF;

    ////

    float16 tmp;
    tmp.data = result;
    return tmp;
}

//================================================================
//
// unpackFloat16
//
//================================================================

float32 unpackFloat16(const float16& value)
{
    uint32 srcUint = * (const uint16*) &value;

    //
    // Decompose src number
    //

    uint32 sign = srcUint >> 15;
    srcUint &= 0x7FFF;

    int32 srcBiasedExponent = srcUint >> 10;
    uint32 srcOriginalMantissa = srcUint & 0x3FF; // 10 bits

    //
    // Dst parts
    //

    int32 dstBiasedExponent = srcBiasedExponent - 0xF + 0x7F;

    uint32 dstMantissa = srcOriginalMantissa;

    //
    // Handle subnormals and zero
    //

    if (srcBiasedExponent == 0)
    {
        if (dstMantissa == 0)
            dstBiasedExponent = 0;
        else
        {
            dstMantissa <<= 1;

            while ((dstMantissa & 0x400) == 0)
            {
                dstMantissa <<= 1;
                dstBiasedExponent--;
            }
        }
    }

    //
    // Shift mantissa
    //

    dstMantissa <<= 13;

    //
    // Handle src INF and NAN
    //

    if (srcBiasedExponent == 0x1F)
    {
        dstBiasedExponent = 0xFF;
        dstMantissa = (srcOriginalMantissa == 0) ? 0 : 0x7FFFFF;
    }

    ////

    union {float32 f32; uint32 u32;} result;

    result.u32 = (sign << 31) | (dstBiasedExponent << 23) | (dstMantissa & 0x7FFFFF);
    return result.f32;
}
