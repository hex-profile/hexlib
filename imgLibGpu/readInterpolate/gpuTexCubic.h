#pragma once

#include <cmath>

#include "vectorTypes/vectorType.h"
#include "gpuDevice/devSampler/devSampler.h"
#include "readInterpolate/cubicCoeffs.h"
#include "numbers/mathIntrinsics.h"

#if DEVCODE

//================================================================
//
// texCubicGrid2D
//
//================================================================

template <typename SamplerType, typename CoeffsFunc>
sysinline typename DevSamplerResult<SamplerType>::T texCubicGrid2D
(
    SamplerType srcSampler,
    float32 X, float32 Y,
    const Point<float32>& srcTexstep,
    CoeffsFunc coeffsFunc
)
{
    float32 bX = floorf(X);
    float32 bY = floorf(Y);

    float32 dX = X - bX;
    float32 dY = Y - bY;

    ////

    Point<float32> mul = srcTexstep;
    Point<float32> add = 0.5f * srcTexstep;

    float32 BX = bX * mul.X + add.X;
    float32 BY = bY * mul.Y + add.Y;

    float32 X0 = BX - 1 * mul.X;
    float32 X1 = BX + 0 * mul.X;
    float32 X2 = BX + 1 * mul.X;
    float32 X3 = BX + 2 * mul.X;

    float32 Y0 = BY - 1 * mul.Y;
    float32 Y1 = BY + 0 * mul.Y;
    float32 Y2 = BY + 1 * mul.Y;
    float32 Y3 = BY + 2 * mul.Y;

    ////

    using VectorFloat = typename DevSamplerResult<SamplerType>::T;

    ////

    float32 CX0, CX1, CX2, CX3;
    coeffsFunc(dX, CX0, CX1, CX2, CX3);

    VectorFloat V0 =
        CX0 * devTex2D(srcSampler, X0, Y0) +
        CX1 * devTex2D(srcSampler, X1, Y0) +
        CX2 * devTex2D(srcSampler, X2, Y0) +
        CX3 * devTex2D(srcSampler, X3, Y0);

    VectorFloat V1 =
        CX0 * devTex2D(srcSampler, X0, Y1) +
        CX1 * devTex2D(srcSampler, X1, Y1) +
        CX2 * devTex2D(srcSampler, X2, Y1) +
        CX3 * devTex2D(srcSampler, X3, Y1);

    VectorFloat V2 =
        CX0 * devTex2D(srcSampler, X0, Y2) +
        CX1 * devTex2D(srcSampler, X1, Y2) +
        CX2 * devTex2D(srcSampler, X2, Y2) +
        CX3 * devTex2D(srcSampler, X3, Y2);

    VectorFloat V3 =
        CX0 * devTex2D(srcSampler, X0, Y3) +
        CX1 * devTex2D(srcSampler, X1, Y3) +
        CX2 * devTex2D(srcSampler, X2, Y3) +
        CX3 * devTex2D(srcSampler, X3, Y3);

    ////

    float32 CY0, CY1, CY2, CY3;
    coeffsFunc(dY, CY0, CY1, CY2, CY3);

    return CY0*V0 + CY1*V1 + CY2*V2 + CY3*V3;
}

//================================================================
//
// texCubic2D
//
//================================================================

template <typename SamplerType>
sysinline typename DevSamplerResult<SamplerType>::T texCubic2D(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
{
    return texCubicGrid2D(srcSampler, pos.X - 0.5f, pos.Y - 0.5f, srcTexstep, cubicCoeffs<float32>);
}

//================================================================
//
// texCubicBspline2D
//
// * The image should be prefiltered.
//
//================================================================

template <typename SamplerType>
sysinline typename DevSamplerResult<SamplerType>::T texCubicBspline2D(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
{
    return texCubicGrid2D(srcSampler, pos.X - 0.5f, pos.Y - 0.5f, srcTexstep, cubicBsplineCoeffs<float32>);
}

//================================================================
//
// texCubicBsplineFast2D
//
// Uses 4 bilinear fetches
//
// * The image should be prefiltered.
// * The image interpolation mode should be set to BILINEAR!
//
//================================================================

template <typename SamplerType>
sysinline typename DevSamplerResult<SamplerType>::T texCubicBsplineFast2D(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
{
    Point<float32> posGrid = pos - 0.5f;

    Point<float32> base = floorf(posGrid);
    Point<float32> frac = posGrid - base;

    ////

    Point<float32> C0, C1, C2, C3;
    cubicBsplineCoeffs(frac.X, C0.X, C1.X, C2.X, C3.X);
    cubicBsplineCoeffs(frac.Y, C0.Y, C1.Y, C2.Y, C3.Y);

    Point<float32> sumA = C0 + C1;
    Point<float32> sumB = C2 + C3;

    Point<float32> divSumA = point(nativeRecip(sumA.X), nativeRecip(sumA.Y));
    Point<float32> divSumB = point(nativeRecip(sumB.X), nativeRecip(sumB.Y));

    Point<float32> posA = (base + (0.5f - 1) + C1 * divSumA) * srcTexstep;
    Point<float32> posB = (base + (0.5f + 1) + C3 * divSumB) * srcTexstep;

    ////

    using VectorFloat = typename DevSamplerResult<SamplerType>::T;

    ////

    VectorFloat vAA = devTex2D(srcSampler, posA.X, posA.Y);
    VectorFloat vAB = devTex2D(srcSampler, posA.X, posB.Y);
    VectorFloat vBA = devTex2D(srcSampler, posB.X, posA.Y);
    VectorFloat vBB = devTex2D(srcSampler, posB.X, posB.Y);

    return linerp2D(sumB, vAA, vAB, vBA, vBB);
}

//----------------------------------------------------------------

#endif
