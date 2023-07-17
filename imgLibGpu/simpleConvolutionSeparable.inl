#pragma once

#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/gpuDevice.h"

//================================================================
//
// simpleConvolutionSeparable
//
//================================================================

template <bool horizontal, Space compressOctaves, typename SamplerType, typename FilterCoeffs>
sysinline auto simpleConvolutionSeparable
(
    SamplerType srcSampler,
    const Point<float32>& srcTexstep,
    const Point<float32>& dstPos,
    const FilterCoeffs& filterCoeffs
)
{
    typedef typename DevSamplerResult<SamplerType>::T FloatType;
    FloatType sum = convertNearest<FloatType>(0);

    ////

    const Space filterSize = COMPILE_ARRAY_SIZE(filterCoeffs);
    COMPILE_ASSERT(!(compressOctaves == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(compressOctaves >= 1) || (filterSize % 2 == 0));

    ////

    const float32 radius = 0.5f * filterSize;

    Point<float32> srcPos = dstPos;

    if (horizontal)
        srcPos.X = float32(1 << compressOctaves) * dstPos.X - (radius - 0.5f);
    else
        srcPos.Y = float32(1 << compressOctaves) * dstPos.Y - (radius - 0.5f);

    ////

    Point<float32> texPos = srcPos * srcTexstep;

    ////

    devUnrollLoop

    for_count (i, filterSize)
    {
        auto value = tex2D(srcSampler, texPos);

        sum += filterCoeffs[i] * value;

        if (horizontal)
            texPos.X += srcTexstep.X;
        else
            texPos.Y += srcTexstep.Y;
    }

    ////

    return sum;
}

//----------------------------------------------------------------

template <Space compressOctaves, typename SamplerType, typename FilterCoeffs>
sysinline auto simpleConvolutionSeparableDynaAxis
(
    SamplerType srcSampler,
    const Point<float32>& srcTexstep,
    const Point<float32>& dstPos,
    const FilterCoeffs& filterCoeffs,
    bool horizontal
)
{
    return
        horizontal ?
        simpleConvolutionSeparable<true, compressOctaves>(srcSampler, srcTexstep, dstPos, filterCoeffs) :
        simpleConvolutionSeparable<false, compressOctaves>(srcSampler, srcTexstep, dstPos, filterCoeffs);
}

//================================================================
//
// simpleConvolutionValueAndSquare
//
//================================================================

template <bool horizontal, Space compressOctaves, typename SamplerType, typename FilterCoeffs>
sysinline void simpleConvolutionValueAndSquare
(
    SamplerType srcSampler,
    const Point<float32>& srcTexstep,
    const Point<float32>& dstPos,
    const FilterCoeffs& filterCoeffs,
    typename DevSamplerResult<SamplerType>::T& resultAvg,
    typename DevSamplerResult<SamplerType>::T& resultAvgSq
)
{
    typedef typename DevSamplerResult<SamplerType>::T FloatType;

    ////

    const Space filterSize = COMPILE_ARRAY_SIZE(filterCoeffs);
    COMPILE_ASSERT(!(compressOctaves == 0) || (filterSize % 2 == 1));
    COMPILE_ASSERT(!(compressOctaves >= 1) || (filterSize % 2 == 0));

    ////

    const float32 radius = 0.5f * filterSize;

    Point<float32> srcPos = dstPos;

    if (horizontal)
        srcPos.X = float32(1 << compressOctaves) * dstPos.X - (radius - 0.5f);
    else
        srcPos.Y = float32(1 << compressOctaves) * dstPos.Y - (radius - 0.5f);

    ////

    Point<float32> texPos = srcPos * srcTexstep;

    ////

    FloatType sum = convertNearest<FloatType>(0);
    FloatType sumSq = convertNearest<FloatType>(0);

    ////

    devUnrollLoop

    for_count (i, filterSize)
    {
        auto value = tex2D(srcSampler, texPos);

        sum += filterCoeffs[i] * value;
        sumSq += filterCoeffs[i] * square(value);

        if (horizontal)
            texPos.X += srcTexstep.X;
        else
            texPos.Y += srcTexstep.Y;
    }

    ////

    resultAvg = sum;
    resultAvgSq = sumSq;
}
