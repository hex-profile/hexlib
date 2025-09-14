#pragma once

#include "rndgen/rndgenBase.h"
#include "numbers/float/floatBase.h"
#include "point/pointBase.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// rndgenUniform
//
// Generates random float in [0, 1) range.
//
//================================================================

sysinline void rndgenUniform(RndgenState& state, uint8& result)
{
    result = rndgen16(state); // 16 random bits
}

//----------------------------------------------------------------

sysinline void rndgenUniform(RndgenState& state, uint16& result)
{
    result = rndgen16(state); // 16 random bits
}

//----------------------------------------------------------------

sysinline void rndgenUniform(RndgenState& state, uint32& result)
{
    result = rndgen32(state);
}

//----------------------------------------------------------------

sysinline void rndgenUniform(RndgenState& state, float32& result)
{
    uint32 tmp = rndgen32(state); // 32 random bits
    result = tmp * 0.2328306436538696e-9f; // div by 2^32, result [0, 1)
}

//----------------------------------------------------------------

sysinline void rndgenUniform(RndgenState& state, Point<float32>& result)
{
    rndgenUniform(state, result.X);
    rndgenUniform(state, result.Y);
}

//================================================================
//
// rndgenUniform<T>
//
//================================================================

template <typename Dst>
sysinline Dst rndgenUniform(RndgenState& state)
{
    Dst dst;
    rndgenUniform(state, dst);
    return dst;
}

//================================================================
//
// rndgenUniformRange
//
//================================================================

template <typename Type>
sysinline Type rndgenUniformRange(RndgenState& state, const Type& a, const Type& b)
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_FLOAT(VECTOR_BASE(Type)));

    return a + (b - a) * rndgenUniform<Type>(state);
}

//================================================================
//
// rndgenUniformFloat
//
// The fastest, should be 3 instructions if used in series.
//
// Generates uniformly distributed float in [0, 1) range,
// not super-quality (low bits of mantissa are less random).
//
// The signed version gives result int [-1/2, +1/2) range.
//
//================================================================

sysinline float32 rndgenUniformFloat(RndgenState& state, float32 factor=1)
{
    rndgenNext(state);
    return state * (factor * (0.2328306437e-9f)); // div by 2^32, result [0, 1)
}

sysinline float32 rndgenUniformSignedFloat(RndgenState& state, float32 factor=1)
{
    rndgenNext(state);
    return float32(int32(state)) * (factor * (0.2328306437e-9f)); // div by 2^32, result [-1/2, +1/2)
}

//================================================================
//
// rndgenQualityGauss
//
// Unit normal random variables (mean = 0 and variance = 1)
// Box-Muller Method
//
//================================================================

sysinline void rndgenQualityGauss(RndgenState& state, float32& r1, float32& r2)
{

retry:

    float32 U1 = rndgenUniformFloat(state);
    float32 U2 = rndgenUniformFloat(state);

    float32 V1 = 2 * U1 - 1; // V1 = [-1, +1]
    float32 V2 = 2 * U2 - 1; // V2 = [-1, +1]

    float32 S = V1*V1 + V2*V2;

    float32 minS = 0.5e-36f; // avoid overflow in (-2*log(S)/S)
    if_not (S >= minS && S < 1) goto retry;

    float32 multiplier = sqrtf(-2 * logf(S) / S);

    r1 = multiplier * V1;
    r2 = multiplier * V2;

}

//================================================================
//
// rndgenGaussApproxThree
//
// By sum of 3 uniformly distributed numbers.
// (mean = 0 and variance = 1)
//
// Gives 9 instructions if used in series, 11 instructions otherwise.
//
//================================================================

template <typename Type>
sysinline Type rndgenGaussApproxThree(RndgenState& state);

//----------------------------------------------------------------

template <>
sysinline float32 rndgenGaussApproxThree(RndgenState& state)
{
    const float32 bias = -3 * 0.5f;
    const float32 factor = 2.f; // compensate for sigma=1

    float32 r0 = rndgenUniformFloat(state, factor);
    float32 r1 = rndgenUniformFloat(state, factor);
    float32 r2 = rndgenUniformFloat(state, factor);

    return (bias * factor) + r0 + r1 + r2;
}

//================================================================
//
// rndgenGaussApproxFour
//
// By sum of 4 uniformly distributed numbers.
// (mean = 0 and variance = 1)
//
// Gives 12 instructions if used in series, 14 instructions otherwise.
//
//================================================================

template <typename Type>
sysinline Type rndgenGaussApproxFour(RndgenState& state);

//----------------------------------------------------------------

template <>
sysinline float32 rndgenGaussApproxFour(RndgenState& state)
{
    const float32 bias = -4 * 0.5f;
    const float32 factor = 1.73205080723775459f; // compensate for sigma=1

    float32 r0 = rndgenUniformFloat(state, factor);
    float32 r1 = rndgenUniformFloat(state, factor);
    float32 r2 = rndgenUniformFloat(state, factor);
    float32 r3 = rndgenUniformFloat(state, factor);

    return (bias * factor) + r0 + r1 + r2 + r3;
}

//================================================================
//
// rndgenLogScale
//
//================================================================

template <typename Type>
sysinline Type rndgenLogScale(RndgenState& state, const Type& minValue, const Type& maxValue)
{
    Type minLog2 = fastLog2(minValue);
    Type maxLog2 = fastLog2(maxValue);

    return fastPow2(rndgenUniformRange(state, minLog2, maxLog2));
}
