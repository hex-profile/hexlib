#pragma once

#include <cmath>

#include "vectorTypes/vectorType.h"
#include "gpuDevice/devSampler/devSampler.h"
#include "readInterpolate/cubicCoeffs.h"
#include "numbers/mathIntrinsics.h"
#include "point/pointMathIntrinsics.h"

//================================================================
//
// Tex2DCubicPreparation
//
//================================================================

struct Tex2DCubicPreparation
{
    float32 X0, X1, X2, X3;
    float32 CX0, CX1, CX2, CX3;

    float32 Y0, Y1, Y2, Y3;
    float32 CY0, CY1, CY2, CY3;
};

//================================================================
//
// tex2DCubicPrepare
//
//================================================================

template <typename CoeffsFunc>
sysinline Tex2DCubicPreparation tex2DCubicPrepare(const Point<float32>& srcPos, const Point<float32>& srcTexstep, CoeffsFunc coeffsFunc)
{
    Tex2DCubicPreparation prep;

    float32 X = srcPos.X - 0.5f;
    float32 Y = srcPos.Y - 0.5f;

    float32 bX = floorf(X);
    float32 bY = floorf(Y);

    float32 dX = X - bX;
    float32 dY = Y - bY;

    ////

    Point<float32> mul = srcTexstep;
    Point<float32> add = 0.5f * srcTexstep;

    float32 BX = bX * mul.X + add.X;
    float32 BY = bY * mul.Y + add.Y;

    prep.X0 = BX - 1 * mul.X;
    prep.X1 = BX + 0 * mul.X;
    prep.X2 = BX + 1 * mul.X;
    prep.X3 = BX + 2 * mul.X;

    prep.Y0 = BY - 1 * mul.Y;
    prep.Y1 = BY + 0 * mul.Y;
    prep.Y2 = BY + 1 * mul.Y;
    prep.Y3 = BY + 2 * mul.Y;

    ////

    coeffsFunc(dX, prep.CX0, prep.CX1, prep.CX2, prep.CX3);
    coeffsFunc(dY, prep.CY0, prep.CY1, prep.CY2, prep.CY3);

    ////

    return prep;
}

//================================================================
//
// tex2DCubicApply
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicApply(SamplerType srcSampler, const Tex2DCubicPreparation& prep)
{
    auto V0 =
        prep.CX0 * devTex2D(srcSampler, prep.X0, prep.Y0) +
        prep.CX1 * devTex2D(srcSampler, prep.X1, prep.Y0) +
        prep.CX2 * devTex2D(srcSampler, prep.X2, prep.Y0) +
        prep.CX3 * devTex2D(srcSampler, prep.X3, prep.Y0);

    auto V1 =
        prep.CX0 * devTex2D(srcSampler, prep.X0, prep.Y1) +
        prep.CX1 * devTex2D(srcSampler, prep.X1, prep.Y1) +
        prep.CX2 * devTex2D(srcSampler, prep.X2, prep.Y1) +
        prep.CX3 * devTex2D(srcSampler, prep.X3, prep.Y1);

    auto V2 =
        prep.CX0 * devTex2D(srcSampler, prep.X0, prep.Y2) +
        prep.CX1 * devTex2D(srcSampler, prep.X1, prep.Y2) +
        prep.CX2 * devTex2D(srcSampler, prep.X2, prep.Y2) +
        prep.CX3 * devTex2D(srcSampler, prep.X3, prep.Y2);

    auto V3 =
        prep.CX0 * devTex2D(srcSampler, prep.X0, prep.Y3) +
        prep.CX1 * devTex2D(srcSampler, prep.X1, prep.Y3) +
        prep.CX2 * devTex2D(srcSampler, prep.X2, prep.Y3) +
        prep.CX3 * devTex2D(srcSampler, prep.X3, prep.Y3);

    ////

    return prep.CY0 * V0 + prep.CY1 * V1 + prep.CY2 * V2 + prep.CY3 * V3;
}

//================================================================
//
// tex2DCubicGeneric
//
//================================================================

template <typename SamplerType, typename CoeffsFunc>
sysinline auto tex2DCubicGeneric
(
    SamplerType srcSampler,
    const Point<float32>& srcPos,
    const Point<float32>& srcTexstep,
    CoeffsFunc coeffsFunc
)
{
    float32 X = srcPos.X - 0.5f;
    float32 Y = srcPos.Y - 0.5f;

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

    float32 CX0, CX1, CX2, CX3;
    coeffsFunc(dX, CX0, CX1, CX2, CX3);

    auto V0 =
        CX0 * devTex2D(srcSampler, X0, Y0) +
        CX1 * devTex2D(srcSampler, X1, Y0) +
        CX2 * devTex2D(srcSampler, X2, Y0) +
        CX3 * devTex2D(srcSampler, X3, Y0);

    auto V1 =
        CX0 * devTex2D(srcSampler, X0, Y1) +
        CX1 * devTex2D(srcSampler, X1, Y1) +
        CX2 * devTex2D(srcSampler, X2, Y1) +
        CX3 * devTex2D(srcSampler, X3, Y1);

    auto V2 =
        CX0 * devTex2D(srcSampler, X0, Y2) +
        CX1 * devTex2D(srcSampler, X1, Y2) +
        CX2 * devTex2D(srcSampler, X2, Y2) +
        CX3 * devTex2D(srcSampler, X3, Y2);

    auto V3 =
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
// tex2DCubic
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubic(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
{
    return tex2DCubicGeneric(srcSampler, pos, srcTexstep, cubicCoeffs<float32>);
}

//================================================================
//
// tex2DCubicBspline
//
// (!) The image should be prefiltered.
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicBspline(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
{
    return tex2DCubicGeneric(srcSampler, pos, srcTexstep, cubicBsplineCoeffs<float32>);
}

//================================================================
//
// tex2DCubicBsplineFast
//
// Uses only 4 bilinear fetches.
//
// (!) The image should be prefiltered.
// (!) The image interpolation mode should be set to BILINEAR.
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicBsplineFast(SamplerType srcSampler, const Point<float32>& pos, const Point<float32>& srcTexstep)
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

    auto vAA = devTex2D(srcSampler, posA.X, posA.Y);
    auto vAB = devTex2D(srcSampler, posA.X, posB.Y);
    auto vBA = devTex2D(srcSampler, posB.X, posA.Y);
    auto vBB = devTex2D(srcSampler, posB.X, posB.Y);

    return linerp2D(sumB, vAA, vAB, vBA, vBB);
}
