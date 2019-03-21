#pragma once

#include "numbers/mathIntrinsics.h"

//================================================================
//
// gaussExpoApprox
//
// The fades from exact 1 to exact 0.
// The function is smooth and monotonic.
//
//================================================================

template <int32 expQuality>
sysinline float32 negExpApprox(float32 x, float32 argScale);


template <>
sysinline float32 negExpApprox<2>(float32 x, float32 argScale)
{
    float32 arg = saturate(1 + (-0.25f * argScale) * x);
    return square(square(arg));
}

template <>
sysinline float32 negExpApprox<3>(float32 x, float32 argScale)
{
    float32 arg = saturate(1 + (-0.125f * argScale) * x);
    return square(square(square(arg)));
}

template <>
sysinline float32 negExpApprox<4>(float32 x, float32 argScale)
{
    float32 arg = saturate(1 + (-0.0625f * argScale) * x);
    return square(square(square(square(arg))));
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// gaussExpoApprox
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <int32 n>
sysinline float32 gaussExpoApprox(float32 x2);

//================================================================
//
// gaussExpoApprox<2>
//
// Takes 5 instructions on GPU.
// The worst error is 4.9 bits.
//
//================================================================

template <>
sysinline float32 gaussExpoApprox<2>(float32 x2)
{
    return negExpApprox<2>(x2, 0.5f * 0.845f);
}

//================================================================
//
// gaussExpoApprox<3>
//
// Takes 6 instructions on GPU.
// The worst error is 5.95 bits.
//
//================================================================

template <>
sysinline float32 gaussExpoApprox<3>(float32 x2)
{
    return negExpApprox<3>(x2, 0.5f * 0.920f);
}

//================================================================
//
// gaussExpoApprox<4>
//
// Takes 7 instructions on GPU.
// The worst error is 6.95 bits.
//
//================================================================

template <>
sysinline float32 gaussExpoApprox<4>(float32 x2)
{
    return negExpApprox<4>(x2, 0.5f * 0.959f);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// sigmoidApprox
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <int expQuality>
sysinline float32 sigmoidApprox(float32 value);

//
// Max error is 5.5 bits
// On Kepler it gives 13 instructions.
//

template <>
sysinline float32 sigmoidApprox<2>(float32 value)
{
    float32 ex = negExpApprox<2>(absv(value), 0.799f);
    float32 result = nativeRecip(1 + ex);
    if (value < 0) result = 1 - result;
    return result;
}

//
// Max error is 6.5 bits
// On Kepler it gives 14 instructions.
//

template <>
sysinline float32 sigmoidApprox<3>(float32 value)
{
    float32 ex = negExpApprox<3>(absv(value), 0.894f);
    float32 result = nativeRecip(1 + ex);
    if (value < 0) result = 1 - result;
    return result;
}

//
// Max error is 7.5 bits.
// On Kepler it gives 15 instructions.
//

template <>
sysinline float32 sigmoidApprox<4>(float32 value)
{
    float32 ex = negExpApprox<4>(absv(value), 0.946f);
    float32 result = nativeRecip(1 + ex);
    if (value < 0) result = 1 - result;
    return result;
}


//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// antiGaussSq
//
// Returns x such that exp(-x^2/2) = value
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <typename Type>
sysinline Type antiGaussSq(const Type& value);

template <>
sysinline float32 antiGaussSq(const float32& value)
{
    float32 limitedValue = clampRange<float32>(value, FLT_MIN, 1.f);
    return (-1.38629436111989062f) * nativeLog2(limitedValue);
}
