#include "convertYuv420ToBgr.h"

#if HOSTCODE
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#endif

#include "convertYuv420/convertYuvRgbFunc.h"
#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "kit/kit.h"
#include "types/lt/ltType.h"
#include "readBordered.h"
#include "imageRead/positionTools.h"

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
static const Space threadCountY = 4;

sysinline Point<Space> threadCount() {return point(Space(threadCountX), Space(threadCountY));}

//================================================================
//
// Filter coeffs
//
//================================================================

#if 1
#define C0 (-3 / 128.f)
#define C1 (-9 / 128.f)
#define C2 (+29 / 128.f)
#define C3 (+111 / 128.f)
#define C4 (+111 / 128.f)
#define C5 (+29 / 128.f)
#define C6 (-9 / 128.f)
#define C7 (-3 / 128.f)
#endif

//================================================================
//
// ConvertParamsYuvBgr
//
//================================================================

template <typename DstPixel>
struct ConvertParamsYuvBgr
{
    Point<float32> lumaTexstep;
    bool chromaIsPacked;
    Point<float32> chromaTexstep;
    Point<Space> srcOffsetDiv2;
    Point<Space> srcSize;
    DstPixel outerColor;
    GpuMatrix<DstPixel> dst;
};

//================================================================
//
// Samplers
//
//================================================================

#if DEVCODE

devDefineSampler(lumaSampler, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(chromaSamplerPacked, DevSampler2D, DevSamplerFloat, 2)
devDefineSampler(chromaSamplerU, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(chromaSamplerV, DevSampler2D, DevSamplerFloat, 1)

#endif

//================================================================
//
// Convert options
//
//================================================================

#define CONVERT_YUV 0x27F51A5A
#define CONVERT_RGB 0x5E597E75

//================================================================
//
// Code: RGB
//
//================================================================

#define DST_PIXEL uint8_x4
#define SUFFIX Bgr_uint8
#define CONVERT CONVERT_RGB

# include "convertYuv420ToBgr.inl"

#undef DST_PIXEL
#undef SUFFIX
#undef CONVERT

//----------------------------------------------------------------

#define DST_PIXEL float16_x4
#define SUFFIX Bgr_float16
#define CONVERT CONVERT_RGB

# include "convertYuv420ToBgr.inl"

#undef DST_PIXEL
#undef SUFFIX
#undef CONVERT

//================================================================
//
// Instances
//
//================================================================

#define MAKE_FUNCTION(baseFunc, SrcPixel, SrcPixel2, DstPixel, suffix) \
    \
    template <> \
    stdbool baseFunc \
    ( \
        const GpuMatrix<const SrcPixel>& srcLuma, \
        const GpuMatrix<const SrcPixel2>& srcChromaPacked, \
        const GpuMatrix<const SrcPixel>& srcChromaU, \
        const GpuMatrix<const SrcPixel>& srcChromaV, \
        const Point<Space>& srcOffset, \
        const DstPixel& outerColor, \
        const GpuMatrix<DstPixel>& dst, \
        stdPars(GpuProcessKit) \
    ) \
    { \
        return baseFunc##suffix(srcLuma, srcChromaPacked, srcChromaU, srcChromaV, srcOffset, outerColor, dst, stdPassThru); \
    }

////

#if HOSTCODE

MAKE_FUNCTION(convertYuv420ToBgr, int8, int8_x2, uint8_x4, _uint8)
MAKE_FUNCTION(convertYuv420ToBgr, float16, float16_x2, uint8_x4, _uint8)
MAKE_FUNCTION(convertYuv420ToBgr, float16, float16_x2, float16_x4, _float16)

#endif
