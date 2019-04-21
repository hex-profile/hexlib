#if HOSTCODE
#include "downsampleTwiceTent.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuSupport/gpuTemplateKernel.h"

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
// Filter coeffs (4 taps == 2 tap tent * downsample 2X)
//
//================================================================

#define C0 0.125f
#define C1 0.375f
#define C2 0.375f
#define C3 0.125f

//================================================================
//
// DownsampleParams
//
//================================================================

template <typename Dst>
struct DownsampleParams
{
    GpuMatrix<Dst> dst;
};

//================================================================
//
// downsampleTwiceKernel
//
//================================================================

#if DEVCODE

template <typename Dst>
inline devDecl void downsampleTwiceKernel(const DownsampleParams<Dst>& o, devPars)
{

    //----------------------------------------------------------------
    //
    // Destination X, Y
    //
    //----------------------------------------------------------------

    Space dstBaseX = devGroupX * threadCountX;
    Space dstBaseY = devGroupY * threadCountY;

    //----------------------------------------------------------------
    //
    // SRAM matrix for source tile: 3 + size + 3
    //
    //----------------------------------------------------------------

    const Space extraL = 1;
    const Space extraR = 1;
    const Space extraLR = extraL + extraR;

    const Space srcSramSizeX = 2 * threadCountX + extraLR;
    const Space srcSramSizeY = 2 * threadCountY + extraLR;

    devSramMatrixDense(srcBuffer, float32, srcSramSizeX, srcSramSizeY);

    #define SRC_BUFFER(X, Y) \
        (MATRIX_ELEMENT(srcBuffer, X, Y))

    //----------------------------------------------------------------
    //
    // Read src tile
    //
    //----------------------------------------------------------------

    #define READ_SRC(X, Y) \
        devTex2D(srcSampler, (X) + 0.5f, (Y) + 0.5f) // index to space coords

    Space srcBaseX = 2 * dstBaseX - extraL;
    Space srcBaseY = 2 * dstBaseY - extraL;

    ////

    COMPILE_ASSERT(threadCountX >= extraLR);
    COMPILE_ASSERT(threadCountY >= extraLR);

    bool extraX = devThreadX < extraLR;
    bool extraY = devThreadY < extraLR;

    ////

    #define READ_ITER(kX, kY) \
        SRC_BUFFER((kX) * threadCountX + devThreadX, (kY) * threadCountY + devThreadY) = \
            READ_SRC(srcBaseX + (kX) * threadCountX + devThreadX, srcBaseY + (kY) * threadCountY + devThreadY)

    READ_ITER(0, 0); READ_ITER(1, 0);
    READ_ITER(0, 1); READ_ITER(1, 1);

    if (extraX) 
    {
        READ_ITER(2, 0); 
        READ_ITER(2, 1);
    }

    if (extraY)
    {
        READ_ITER(0, 2); 
        READ_ITER(1, 2);
    }

    if (extraX && extraY)
        READ_ITER(2, 2);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Downsample Y
    //
    //----------------------------------------------------------------

    devSramMatrixDense(tmpBuffer, float32, srcSramSizeX, threadCountY);

    #define TMP_BUFFER(X, Y) \
        (MATRIX_ELEMENT(tmpBuffer, X, Y))

    ////

    Space bY = 2 * devThreadY;

    #define DOWNSAMPLE_VERTICAL(bX) \
        TMP_BUFFER(bX, devThreadY) = \
            C0 * SRC_BUFFER(bX, bY + 0) + \
            C1 * SRC_BUFFER(bX, bY + 1) + \
            C2 * SRC_BUFFER(bX, bY + 2) + \
            C3 * SRC_BUFFER(bX, bY + 3) 

    DOWNSAMPLE_VERTICAL(devThreadX + 0 * threadCountX);
    DOWNSAMPLE_VERTICAL(devThreadX + 1 * threadCountX);

    if (extraX)
    DOWNSAMPLE_VERTICAL(devThreadX + 2 * threadCountX);

    ////

    devSyncThreads();

    //----------------------------------------------------------------
    //
    // Downsample X
    //
    //----------------------------------------------------------------

    Space bX = 2 * devThreadX;

    float32 result = 
        C0 * TMP_BUFFER(bX + 0, devThreadY) +
        C1 * TMP_BUFFER(bX + 1, devThreadY) +
        C2 * TMP_BUFFER(bX + 2, devThreadY) +
        C3 * TMP_BUFFER(bX + 3, devThreadY);

    //----------------------------------------------------------------
    //
    // Write output
    //
    //----------------------------------------------------------------

    MATRIX_EXPOSE_EX(o.dst, dst);

    Space dstX = dstBaseX + devThreadX;
    Space dstY = dstBaseY + devThreadY;

    if (dstX < dstSizeX && dstY < dstSizeY)
        storeNorm(MATRIX_POINTER(dst, dstX, dstY), result);
}

#endif

//================================================================
//
// Downsample kernels
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), downsampleTwiceKernel, DownsampleParams, downsampleTwiceKernelLink, dowdownsampleTwiceKernel8u, (uint8))
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), downsampleTwiceKernel, DownsampleParams, downsampleTwiceKernelLink, downsampleTwiceKernel16f, (float16))

//================================================================
//
// downsampleTwiceTent
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst>
stdbool downsampleTwiceTent(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true;

    ////

    require(kit.gpuSamplerSetting.setSamplerImage(srcSampler, src, BORDER_CLAMP, false, true, false, stdPass));

    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            divUpNonneg(dst.size(), point(threadCountX, threadCountY)),
            point(threadCountX, threadCountY),
            areaOf(dst),
            downsampleTwiceKernelLink<Dst>(),
            DownsampleParams<Dst>{dst},
            kit.gpuCurrentStream,
            stdPass
        )
    );

    ////

    stdEnd;
}

#endif

//================================================================
//
// Instantiations
//
//================================================================

#if HOSTCODE

INSTANTIATE_FUNC((downsampleTwiceTent<uint8, uint8>))
INSTANTIATE_FUNC((downsampleTwiceTent<float16, float16>))

#endif
