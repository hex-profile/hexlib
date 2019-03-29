#if HOSTCODE
#include "upsampleTwiceTent.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#include "dataAlloc/gpuMatrixMemory.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "readBordered.h"

//================================================================
//
// threadCountX
// threadCountY
//
// Each thread computes one destination pixel.
// So destination tile is (threadCountX, threadCountY)
//
//================================================================

static const Space threadCountX = 32;
static const Space threadCountY = 8;

//================================================================
//
// srcSampler
//
//================================================================

devDefineSampler(srcSampler, DevSampler2D, DevSamplerFloat, 1)

//================================================================
//
// UpsampleParams
//
//================================================================

struct UpsampleParams
{
    GpuMatrix<uint8> dst;
};

//================================================================
//
// upsample2KernelHorizontal
//
//================================================================

#if DEVCODE

devDefineKernel(upsample2KernelTent, UpsampleParams, o)
{
    MATRIX_EXPOSE_EX(o.dst, dst);

    Space dstX = devGroupX * threadCountX + devThreadX;
    Space dstY = devGroupY * threadCountY + devThreadY;

    float32 dstXs = dstX + 0.5f;
    float32 dstYs = dstY + 0.5f;

    float32 srcXs = 0.5f * dstXs;
    float32 srcYs = 0.5f * dstYs;

    float32 value = devTex2D(srcSampler, srcXs, srcYs);

    if (MATRIX_VALID_ACCESS(dst, dstX, dstY))
        storeNorm(MATRIX_POINTER(dst, dstX, dstY), value);
}

#endif

//================================================================
//
// upsampleTwiceTent
//
//================================================================

#if HOSTCODE

bool upsampleTwiceTent(const GpuMatrix<const uint8>& src, const GpuMatrix<uint8>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    ////

    if (kit.dataProcessing)
        require(kit.gpuSamplerSetting.setSamplerImage(srcSampler, src, BORDER_CLAMP, true, true, false, stdPass));

    ////

    if (kit.dataProcessing)
    {
        require
        (
            kit.gpuKernelCalling.callKernel
            (
                divUpNonneg(dst.size(), point(threadCountX, threadCountY)),
                point(threadCountX, threadCountY),
                areaOf(dst),
                upsample2KernelTent,
                UpsampleParams{dst},
                kit.gpuCurrentStream,
                stdPass
            )
        );
    }

    ////

    stdEnd;
}

#endif
