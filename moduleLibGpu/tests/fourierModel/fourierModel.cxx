#if HOSTCODE
#include "fourierModel.h"
#endif

#include "numbers/float/floatType.h"
#include "numbers/mathIntrinsics.h"
#include "gpuSupport/gpuTool.h"
#include "vectorTypes/vectorOperations.h"
#include "vectorTypes/vectorType.h"
#include "mathFuncs/rotationMath.h"
#include "types/lt/ltType.h"

//================================================================
//
// fourierSeparableFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    fourierSeparableFunc,
    ((const float32_x2, src, INTERP_NONE, BORDER_ZERO))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    ((float32_x2, dst)),
    ((Space, srcSize))
    ((LinearTransform<float32>, dstToFreq))
    ((bool, horizontal))
)
#if DEVCODE
{

    float32 freqPos = horizontal ? Xs : Ys;
    float32 freq = dstToFreq(freqPos);

    float32 srcSizef = convertFloat32(srcSize);
    float32 srcCenter = 0.5f * srcSizef;

    ////

    Space iX = 0; Space iY = 0;
    Space dX = 0; Space dY = 0;

    if (horizontal)
        {dX = 1; iY = Y;}
    else
        {dY = 1; iX = X;}

    //// 

    auto sum = zeroOf<float32_x2>();

    for_count (i, srcSize)
    {
        auto value = devTex2D(srcSampler, (iX + 0.5f) * srcTexstep.X, (iY + 0.5f) * srcTexstep.Y);

        float32 s = (i + 0.5f) - srcCenter;
        float32 phase = s * freq;
        auto spiral = circleCcw(phase);

        sum += complexMul(spiral, value);

        iX += dX;
        iY += dY;
    }

    ////

    *dst = sum / srcSizef;
}
#endif
GPUTOOL_2D_END;

//================================================================
//
// fourierSeparable
//
//================================================================

#if HOSTCODE

stdbool fourierSeparable
(
    const GpuMatrix<const float32_x2>& src, 
    const GpuMatrix<float32_x2>& dst,
    const Point<float32>& minPeriod, 
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
)
{
    REQUIRE(minPeriod > 0);

    ////

    Point<float32> dstSizef = convertFloat32(dst.size());
    LinearTransform<float32> dstToFreqX = ltByTwoPoints(0.f, -1.f/minPeriod.X, dstSizef.X, +1.f/minPeriod.X);
    LinearTransform<float32> dstToFreqY = ltByTwoPoints(0.f, -1.f/minPeriod.Y, dstSizef.Y, +1.f/minPeriod.Y);

    ////

    GPU_MATRIX_ALLOC(tmp, float32_x2, point(dst.sizeX(), src.sizeY()));
    require(fourierSeparableFunc(src, circleTable, tmp, src.sizeX(), dstToFreqX, true, stdPass));
    require(fourierSeparableFunc(tmp, circleTable, dst, src.sizeY(), dstToFreqY, false, stdPass));

    returnTrue;
}

#endif

//================================================================
//
// invFourierSeparableFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    invFourierSeparableFunc,
    ((const float32_x2, src, INTERP_NONE, BORDER_ZERO))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    ((float32_x2, dst)),
    ((Space, srcSize))
    ((Space, dstSize))
    ((LinearTransform<float32>, srcToFreq))
    ((bool, horizontal))
    ((bool, normalize))
)
#if DEVCODE
{

    float32 srcSizef = convertFloat32(srcSize);
    //float32 srcCenter = 0.5f * srcSizef;

    float32 dstSizef = convertFloat32(dstSize);
    float32 dstCenter = 0.5f * dstSizef;

    ////

    Space iX = 0; Space iY = 0;
    Space dX = 0; Space dY = 0;

    if (horizontal)
        {dX = 1; iY = Y;}
    else
        {dY = 1; iX = X;}

    //// 

    float32 spatialPos = (horizontal ? Xs : Ys);

    auto sum = zeroOf<float32_x2>();

    ////

    for_count (f, srcSize)
    {
        auto value = devTex2D(srcSampler, (iX + 0.5f) * srcTexstep.X, (iY + 0.5f) * srcTexstep.Y);

        float32 freq = srcToFreq(f + 0.5f);

        float32 phase = -(spatialPos - dstCenter) * freq;
        auto spiral = circleCcw(phase);

        sum += complexMul(spiral, value);

        iX += dX;
        iY += dY;
    }

    ////

    if (normalize) sum /= srcSizef;

    *dst = sum;
}
#endif
GPUTOOL_2D_END;

//================================================================
//
// invFourierSeparable
//
//================================================================

#if HOSTCODE

stdbool invFourierSeparable
(
    const GpuMatrix<const float32_x2>& src, 
    const GpuMatrix<float32_x2>& dst,
    const Point<float32>& minPeriod, 
    const GpuMatrix<const float32_x2>& circleTable,
    bool normalize,
    stdPars(GpuProcessKit)
)
{
    Point<float32> srcSizef = convertFloat32(src.size());
    LinearTransform<float32> srcToFreqX = ltByTwoPoints(0.f, -1.f/minPeriod.X, srcSizef.X, +1.f/minPeriod.X);
    LinearTransform<float32> srcToFreqY = ltByTwoPoints(0.f, -1.f/minPeriod.Y, srcSizef.Y, +1.f/minPeriod.Y);

    ////

    GPU_MATRIX_ALLOC(tmp, float32_x2, point(dst.sizeX(), src.sizeY()));
    require(invFourierSeparableFunc(src, circleTable, tmp, src.sizeX(), dst.sizeX(), srcToFreqX, true, normalize, stdPass));
    require(invFourierSeparableFunc(tmp, circleTable, dst, src.sizeY(), dst.sizeY(), srcToFreqY, false, normalize, stdPass));

    returnTrue;
}

#endif

//================================================================
//
// orientedFourierKernel
//
//================================================================

GPUTOOL_2D_BEG
(
    orientedFourierKernel,
    ((const float32_x2, src, INTERP_NONE, BORDER_ZERO))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    ((float32_x2, dst)),
    ((Point<Space>, srcSize))
    ((LinearTransform<float32>, dstToFreq))
)
#if DEVCODE
{

    float32 freq = dstToFreq(Xs);

    ////

    float32 orientAngle = Ys / vGlobSize.Y; // (0..1)
    auto orientBackRotate = complexConjugate(devTex2D(circleTableSampler, 0.5f * orientAngle, 0)); // (0, 1/2)

    ////

    auto srcCenter = 0.5f * convertNearest<float32_x2>(srcSize);

    //// 

    auto sum = zeroOf<float32_x2>();

    for_count (iY, srcSize.Y)
    {
        for_count (iX, srcSize.X)
        {
            float32 srcX = iX + 0.5f;
            float32 srcY = iY + 0.5f;

            auto value = devTex2D(srcSampler, srcX * srcTexstep.X, srcY * srcTexstep.Y);

            float32 s = complexMul(make_float32_x2(srcX, srcY) - srcCenter, orientBackRotate).x;
            float32 phase = -s * freq;
            auto spiral = devTex2D(circleTableSampler, phase, 0);
            sum += complexMul(spiral, value);
        }
    }

    ////

    *dst = sum / float32(srcSize.X * srcSize.Y);
}
#endif
GPUTOOL_2D_END;

//================================================================
//
// orientedFourier
//
//================================================================

#if HOSTCODE

stdbool orientedFourier
(
    const GpuMatrix<const float32_x2>& src, 
    const GpuMatrix<float32_x2>& dst,
    float32 minPeriod, 
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
)
{
    REQUIRE(minPeriod > 0);

    float32 dstSizef = convertFloat32(dst.sizeX());
    LinearTransform<float32> dstToFreqX = ltByTwoPoints(0.f, -1.f/minPeriod, dstSizef, +1.f/minPeriod);

    require(orientedFourierKernel(src, circleTable, dst, src.size(), dstToFreqX, stdPass));

    returnTrue;
}

#endif

//================================================================
//
// invOrientedFourierKernel
//
//================================================================

GPUTOOL_2D_BEG
(
    invOrientedFourierKernel,
    ((const float32_x2, freq, INTERP_NONE, BORDER_ZERO))
    ((const float32_x2, circleTable, INTERP_LINEAR, BORDER_WRAP)),
    ((float32_x2, dst)),
    ((Point<Space>, freqSize))
    ((LinearTransform<float32>, srcToFreq))
)
#if DEVCODE
{
    float32 divOrientCount = 1.f / freqSize.Y;
    auto dstCenter = 0.5f * convertNearest<float32_x2>(vGlobSize);

    //// 

    auto sum = zeroOf<float32_x2>();

    for_count (iY, freqSize.Y)
    {
        float32 orientAngle = (iY + 0.5f) * divOrientCount; // (0, 1)
        auto orientBackRotate = complexConjugate(devTex2D(circleTableSampler, 0.5f * orientAngle, 0)); // (0, 1/2)
        auto pos = complexMul(make_float32_x2(Xs, Ys) - dstCenter, orientBackRotate);

        for_count (iX, freqSize.X)
        {
            float32 frX = iX + 0.5f;
            float32 frY = iY + 0.5f;

            auto freqResponse = devTex2D(freqSampler, frX * freqTexstep.X, frY * freqTexstep.Y);
            VECTOR_DECOMPOSE(freqResponse);

            float32 freq = srcToFreq(iX + 0.5f);
            float32 phase = pos.x * freq;
            auto spiral = devTex2D(circleTableSampler, phase, 0);
            sum += complexMul(spiral, freqResponse);
        }
    }

    ////

    *dst = sum;
}
#endif
GPUTOOL_2D_END;

//================================================================
//
// invOrientedFourier
//
//================================================================

#if HOSTCODE

stdbool invOrientedFourier
(
    const GpuMatrix<const float32_x2>& src, 
    const GpuMatrix<float32_x2>& dst,
    float32 minPeriod, 
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
)
{
    REQUIRE(minPeriod > 0);
    Point<float32> srcSizef = convertFloat32(src.size());
    LinearTransform<float32> srcToFreqX = ltByTwoPoints(0.f, -1.f/minPeriod, srcSizef.X, +1.f/minPeriod);

    require(invOrientedFourierKernel(src, circleTable, dst, src.size(), srcToFreqX, stdPass));

    returnTrue;
}

#endif
