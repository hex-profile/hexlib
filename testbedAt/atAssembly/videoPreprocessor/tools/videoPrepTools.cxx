#if HOSTCODE
#include "videoPrepTools.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "mathFuncs/rotationMath.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "vectorTypes/vectorOperations.h"
#include "gpuSupport/gpuTexTools.h"
#include "flipMatrix.h"
#include "numbers/divRound.h"
#include "rndgen/rndgenFloat.h"
#include "gpuDevice/loadstore/loadNorm.h"

//================================================================
//
// CopyRectKernel
//
//================================================================

struct CopyRectKernel
{
    Point<float32> ofsPlusHalf;
    GpuMatrix<uint32> dst;
};

//----------------------------------------------------------------

devDefineSampler(srcSamplerBgra, DevSampler2D, DevSamplerUint, 1)

//================================================================
//
// copyRectKernel
//
//================================================================

const Space copyThreadCountX = 64;

//----------------------------------------------------------------

#if DEVCODE 

devDefineKernel(copyRectKernel, CopyRectKernel, o)
{
    MATRIX_EXPOSE_EX(o.dst, dst);
 
    Space X = devGroupX * copyThreadCountX + devThreadX;
    Space Y = devGroupY;
 
    if_not (X < dstSizeX) return;
 
    float32 srcXs = X + o.ofsPlusHalf.X;
    float32 srcYs = Y + o.ofsPlusHalf.Y;

    auto srcValue = devTex2D(srcSamplerBgra, srcXs, srcYs);

    MATRIX_ELEMENT(dst, X, Y) = srcValue;
}

#endif

//================================================================
//
// copyImageRect
//
//================================================================

#if HOSTCODE

stdbool copyImageRect(const GpuMatrix<const uint8_x4>& src, const Point<Space>& ofs, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit))
{
    if_not (kit.dataProcessing)
        returnTrue;

    ////

    GpuMatrix<const uint8_x4> tmpSrc = src;
    GpuMatrix<uint8_x4> tmpDst = dst;
    Point<Space> tmpOfs = ofs;

    ////

    if (tmpSrc.memPitch() < 0)
    {
        tmpSrc = flipMatrix(tmpSrc);
        tmpDst = flipMatrix(tmpDst);
        tmpOfs.Y = src.sizeY() - dst.sizeY() - tmpOfs.Y;
    }

    ////

    require(kit.gpuSamplerSetting.setSamplerImage(srcSamplerBgra, recastElement<const uint32>(tmpSrc), BORDER_ZERO, false, false, false, stdPass));

    ////

    CopyRectKernel params;
    params.ofsPlusHalf = convertFloat32(tmpOfs) + 0.5f;
    params.dst = recastElement<uint32>(tmpDst);

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(tmpDst.size(), point(copyThreadCountX, 1)),
            point(copyThreadCountX, 1),
            areaOf(tmpDst),
            copyRectKernel,
            params,
            kit.gpuCurrentStream,
            stdPass
        )
    );

    returnTrue;
}

#endif

//================================================================
//
// generateGrating
//
//================================================================

GPUTOOL_2D
(
    generateGrating,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((float32, period))
    ((Point<float32>, transMul)) ((Point<float32>, transAdd))
    ((bool, rectangleShape)),

    {
        Point<float32> srcPos = complexMul(point(Xs, Ys), transMul) + transAdd;

        float32 divPeriod = 1.f / period;
        float32 value = circleCCW(srcPos.X * divPeriod).X;
        if (rectangleShape) value = (value >= 0) ? +1.f : -1.f;

        storeNorm(dst, vectorExtend<float32_x4>(0.5f * (value + 1)));
    }
)

//================================================================
//
// generatePulse
//
//================================================================

GPUTOOL_2D
(
    generatePulse,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((Point<Space>, ofs))
    ((Space, period)),

    {
        Point<Space> pos = point(X, Y) + ofs - (vGlobSize >> 1);

        bool condX = pos.X == divDown(pos.X, period) * period;
        bool condY = pos.Y == divDown(pos.Y, period) * period;

        float32 value = condX && condY;
        storeNorm(dst, vectorExtend<float32_x4>(value));
    }
)

//================================================================
//
// generateRandom
//
//================================================================

GPUTOOL_2D
(
    generateRandom,
    PREP_EMPTY,
    ((uint8_x4, dst))
    ((RndgenState, rndgenState)),
    PREP_EMPTY,

    {
        RndgenState rnd = *rndgenState;
        float32 value = rndgenUniformFloat(rnd);
        *rndgenState = rnd;

        storeNorm(dst, vectorExtend<float32_x4>(value));
    }
)

//================================================================
//
// generateAdditionalGaussNoise
//
//================================================================

GPUTOOL_2D
(
    generateAdditionalGaussNoise,
    PREP_EMPTY,
    ((const uint8_x4, src))
    ((uint8_x4, dst))
    ((RndgenState, rndgenState)),
    ((float32, sigma)),

    {
        auto srcValue = loadNorm(src);

        RndgenState rnd = *rndgenState;

        ////

        auto err = zeroOf<float32_x4>();
        rndgenQualityGauss(rnd, err.x, err.y);
        rndgenQualityGauss(rnd, err.z, err.w);

        auto dstValue = srcValue + sigma * err;

        ////

        *rndgenState = rnd;

        storeNorm(dst, dstValue);
    }
)
