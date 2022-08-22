#pragma once

#include "vectorTypes/vectorType.h"
#include "gpuDevice/devSampler/devSampler.h"
#include "readInterpolate/cubicCoeffs.h"
#include "numbers/mathIntrinsics.h"
#include "point/pointFunctions.h"

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
sysinline Tex2DCubicPreparation tex2DCubicPrepare(const Point<float32>& pos, const Point<float32>& texstep)
{
    Tex2DCubicPreparation prep;

    float32 X = pos.X - 0.5f;
    float32 Y = pos.Y - 0.5f;

    float32 bX = floorv(X);
    float32 bY = floorv(Y);

    float32 dX = X - bX;
    float32 dY = Y - bY;

    ////

    Point<float32> mul = texstep;
    Point<float32> add = 0.5f * texstep;

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

    CoeffsFunc::func(dX, prep.CX0, prep.CX1, prep.CX2, prep.CX3);
    CoeffsFunc::func(dY, prep.CY0, prep.CY1, prep.CY2, prep.CY3);

    ////

    return prep;
}

//================================================================
//
// tex2DCubicApply
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicApply(SamplerType sampler, const Tex2DCubicPreparation& prep)
{
    auto V0 =
        prep.CX0 * devTex2D(sampler, prep.X0, prep.Y0) +
        prep.CX1 * devTex2D(sampler, prep.X1, prep.Y0) +
        prep.CX2 * devTex2D(sampler, prep.X2, prep.Y0) +
        prep.CX3 * devTex2D(sampler, prep.X3, prep.Y0);

    auto V1 =
        prep.CX0 * devTex2D(sampler, prep.X0, prep.Y1) +
        prep.CX1 * devTex2D(sampler, prep.X1, prep.Y1) +
        prep.CX2 * devTex2D(sampler, prep.X2, prep.Y1) +
        prep.CX3 * devTex2D(sampler, prep.X3, prep.Y1);

    auto V2 =
        prep.CX0 * devTex2D(sampler, prep.X0, prep.Y2) +
        prep.CX1 * devTex2D(sampler, prep.X1, prep.Y2) +
        prep.CX2 * devTex2D(sampler, prep.X2, prep.Y2) +
        prep.CX3 * devTex2D(sampler, prep.X3, prep.Y2);

    auto V3 =
        prep.CX0 * devTex2D(sampler, prep.X0, prep.Y3) +
        prep.CX1 * devTex2D(sampler, prep.X1, prep.Y3) +
        prep.CX2 * devTex2D(sampler, prep.X2, prep.Y3) +
        prep.CX3 * devTex2D(sampler, prep.X3, prep.Y3);

    ////

    return prep.CY0 * V0 + prep.CY1 * V1 + prep.CY2 * V2 + prep.CY3 * V3;
}

//================================================================
//
// tex2DCubicGeneric
//
//================================================================

template <typename CoeffsFunc, typename SamplerType>
sysinline auto tex2DCubicGeneric(SamplerType sampler, const Point<float32>& pos, const Point<float32>& texstep)
{
    float32 X = pos.X - 0.5f;
    float32 Y = pos.Y - 0.5f;

    float32 bX = floorv(X);
    float32 bY = floorv(Y);

    float32 dX = X - bX;
    float32 dY = Y - bY;

    ////

    Point<float32> mul = texstep;
    Point<float32> add = 0.5f * texstep;

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
    CoeffsFunc::func(dX, CX0, CX1, CX2, CX3);

    auto V0 =
        CX0 * devTex2D(sampler, X0, Y0) +
        CX1 * devTex2D(sampler, X1, Y0) +
        CX2 * devTex2D(sampler, X2, Y0) +
        CX3 * devTex2D(sampler, X3, Y0);

    auto V1 =
        CX0 * devTex2D(sampler, X0, Y1) +
        CX1 * devTex2D(sampler, X1, Y1) +
        CX2 * devTex2D(sampler, X2, Y1) +
        CX3 * devTex2D(sampler, X3, Y1);

    auto V2 =
        CX0 * devTex2D(sampler, X0, Y2) +
        CX1 * devTex2D(sampler, X1, Y2) +
        CX2 * devTex2D(sampler, X2, Y2) +
        CX3 * devTex2D(sampler, X3, Y2);

    auto V3 =
        CX0 * devTex2D(sampler, X0, Y3) +
        CX1 * devTex2D(sampler, X1, Y3) +
        CX2 * devTex2D(sampler, X2, Y3) +
        CX3 * devTex2D(sampler, X3, Y3);

    ////

    float32 CY0, CY1, CY2, CY3;
    CoeffsFunc::func(dY, CY0, CY1, CY2, CY3);

    return CY0*V0 + CY1*V1 + CY2*V2 + CY3*V3;
}

//================================================================
//
// tex2DCubic
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubic(SamplerType sampler, const Point<float32>& pos, const Point<float32>& texstep)
{
    return tex2DCubicGeneric<CubicCoeffs>(sampler, pos, texstep);
}

//================================================================
//
// tex2DCubicBspline
//
// (!) The image should be prefiltered.
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicBspline(SamplerType sampler, const Point<float32>& pos, const Point<float32>& texstep)
{
    return tex2DCubicGeneric<CubicBsplineCoeffs>(sampler, pos, texstep);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// tex2DCubicBsplineFast
//
// Uses only 4 bilinear fetches.
//
// (!) The image should be prefiltered.
// (!) The image interpolation mode should be set to BILINEAR.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

struct Tex2DCubicBsplineFastPreparation
{
    Point<float32> posA;
    Point<float32> posB;
    Point<float32> sumB;
};

//================================================================
//
// tex2DCubicBsplineFastPrepare
//
//================================================================

sysinline auto tex2DCubicBsplineFastPrepare(const Point<float32>& pos, const Point<float32>& texstep)
{
    Tex2DCubicBsplineFastPreparation prep;

    Point<float32> posGrid = pos - 0.5f;

    Point<float32> base = floorv(posGrid);
    Point<float32> frac = posGrid - base;

    ////

    Point<float32> C0, C1, C2, C3;
    CubicBsplineCoeffs::func(frac.X, C0.X, C1.X, C2.X, C3.X);
    CubicBsplineCoeffs::func(frac.Y, C0.Y, C1.Y, C2.Y, C3.Y);

    Point<float32> sumA = C0 + C1;
    Point<float32> sumB = C2 + C3;

    Point<float32> divSumA = point(fastRecip(sumA.X), fastRecip(sumA.Y));
    Point<float32> divSumB = point(fastRecip(sumB.X), fastRecip(sumB.Y));

    prep.posA = (base + (0.5f - 1) + C1 * divSumA) * texstep;
    prep.posB = (base + (0.5f + 1) + C3 * divSumB) * texstep;

    prep.sumB = sumB;

    return prep;
}

//================================================================
//
// tex2DCubicBsplineFastApply
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicBsplineFastApply(SamplerType sampler, const Tex2DCubicBsplineFastPreparation& prep)
{
    auto vAA = devTex2D(sampler, prep.posA.X, prep.posA.Y);
    auto vAB = devTex2D(sampler, prep.posA.X, prep.posB.Y);
    auto vBA = devTex2D(sampler, prep.posB.X, prep.posA.Y);
    auto vBB = devTex2D(sampler, prep.posB.X, prep.posB.Y);

    return linerp2D(prep.sumB, vAA, vAB, vBA, vBB);
}

//================================================================
//
// tex2DCubicBsplineFast
//
//================================================================

template <typename SamplerType>
sysinline auto tex2DCubicBsplineFast(SamplerType sampler, const Point<float32>& pos, const Point<float32>& texstep)
{
    auto prep = tex2DCubicBsplineFastPrepare(pos, texstep);
    return tex2DCubicBsplineFastApply(sampler, prep);
}
