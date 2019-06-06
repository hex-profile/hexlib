#if HOSTCODE
#include "convertBgr32ToYuv420.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "flipMatrix.h"
#include "errorLog/errorLog.h"
#endif

#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "convertYuv420/convertYuvRgbFunc.h"
#include "types/lt/ltType.h"
#include "gpuSupport/gpuTexTools.h"

//================================================================
//
// Downsampling 2X in SPACE coordinates.
// 
// Compute kth dst element.
// Xk = k + 0.5 ; to space coord
// Xi = 2 Xk = 2k + 1 ; space in src
// I = Xi - 0.5 = 2k + 0.5
//
// Filter kernel continuous support is [-2, +2] in dst space,
// in src space it will be [-4, +4].
//
// Iminf = 2k + 0.5 - 4
// Imaxf = 2k + 0.5 + 4
// 
// Compute range of covered integer indices:
// Imin = ceil(2k + 0.5 - 4) = 2k - 4 + ceil(0.5) = 2k - 4 + 1 = 2k - 3
// Imax = floor(2k + 0.5 + 4) = 2k + 4 + floor(0.5) = 2k + 4
//
// Kernel size is Imax-Imin+1 = 8 elements.
// First element is positioned at 2k - 3 src element for kth dst element.
//
//================================================================

//================================================================
//
// threadCountX
//
//================================================================

static const Space threadCountX = 32;
static const Space threadCountY = 16;

//================================================================
//
// ConvertBgrYuv420Params
//
//================================================================

template <typename DstPixel, typename DstPixel2>
struct ConvertBgrYuv420Params
{
    LinearTransform<Point<float32>> srcTransform;

    GpuMatrix<DstPixel> dstLuma;
    GpuMatrix<DstPixel2> dstChroma;
    GpuMatrix<DstPixel> dstChromaU;
    GpuMatrix<DstPixel> dstChromaV;
};

//================================================================
//
// srcSampler
//
//================================================================

devDefineSampler(srcSampler, DevSampler2D, DevSamplerFloat, 4)

//================================================================
//
// Filter coeffs (8 taps == 4 tap cubic * downsample 2X)
//
//================================================================

#define C0 (-3 / 256.f)
#define C1 (-9 / 256.f)
#define C2 (+29 / 256.f)
#define C3 (+111 / 256.f)
#define C4 (+111 / 256.f)
#define C5 (+29 / 256.f)
#define C6 (-9 / 256.f)
#define C7 (-3 / 256.f)

//================================================================
//
// convertBgr32ToYuv420.inl
//
//================================================================

#define DST_PIXEL int8
#define DST_PIXEL2 int8_x2
# include "convertBgr32ToYuv420.inl"
#undef DST_PIXEL
#undef DST_PIXEL2

//----------------------------------------------------------------

#define DST_PIXEL float16
#define DST_PIXEL2 float16_x2
# include "convertBgr32ToYuv420.inl"
#undef DST_PIXEL
#undef DST_PIXEL2
