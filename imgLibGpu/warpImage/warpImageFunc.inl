#include "warpImage.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"
#include "bsplinePrefilter/bsplinePrefilterSettings.h"
#include "bsplinePrefilter/bsplinePrefilter.h"
#include "types/lt/ltType.h"

#if HOSTCODE
#include "dataAlloc/gpuMatrixMemory.h"
#endif

//================================================================
//
// SRC_PIXEL
// DST_PIXEL
//
//================================================================

#define SRC_PIXEL PREP_PASS(PREP_ARG2_0 PIXELS)
#define DST_PIXEL PREP_PASS(PREP_ARG2_1 PIXELS)

//================================================================
//
// warpImageFunc
//
//================================================================

#define TMP_MACRO(interpMode, borderMode, texInterpolation, texStatement) \
    \
    GPUTOOL_2D \
    ( \
        PREP_PASTE_UNDER5(warpImageFunc, SRC_PIXEL, DST_PIXEL, interpMode, borderMode), \
        ((const SRC_PIXEL, src, texInterpolation, borderMode)) \
        ((const float32_x2, map, INTERP_LINEAR, BORDER_CLAMP)), \
        ((DST_PIXEL, dst)), \
        ((LinearTransform<Point<float32>>, srcTransform)) \
        ((Point<float32>, mapScaleFactor)) \
        ((Point<float32>, mapValueFactor)), \
        \
        { \
            Point<float32> pos = point(Xs, Ys); \
            auto offset = tex2D(mapSampler, pos * mapScaleFactor * mapTexstep); \
            Point<float32> srcPos = srcTransform(pos + mapValueFactor * point(offset.x, offset.y)); \
            storeNorm(dst, texStatement); \
        } \
    )

#define TMP_MACRO2(borderMode) \
    TMP_MACRO(INTERP_LINEAR, borderMode, INTERP_LINEAR, tex2D(srcSampler, srcPos * srcTexstep)) \
    TMP_MACRO(INTERP_CUBIC, borderMode, INTERP_NONE, tex2DCubic(srcSampler, srcPos, srcTexstep)) \
    TMP_MACRO(INTERP_CUBIC_BSPLINE, borderMode, INTERP_LINEAR, tex2DCubicBsplineFast(srcSampler, srcPos, srcTexstep)) \

TMP_MACRO2(BORDER_ZERO)
TMP_MACRO2(BORDER_MIRROR)

#undef TMP_MACRO
#undef TMP_MACRO2

//================================================================
//
// warpImage
//
//================================================================

#if HOSTCODE

template <>
stdbool warpImage
(
    const GpuMatrix<const SRC_PIXEL>& src,
    const LinearTransform<Point<float32>>& srcTransform,
    const GpuMatrix<const float32_x2>& map,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    BorderMode borderMode,
    InterpType interpType,
    const GpuMatrix<DST_PIXEL> dst,
    stdPars(GpuProcessKit)
)
{
    #define TMP_MACRO(borderMode) \
        { \
            if (interpType == INTERP_LINEAR) \
                require(PREP_PASTE_UNDER5(warpImageFunc, SRC_PIXEL, DST_PIXEL, INTERP_LINEAR, borderMode)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \
            else if (interpType == INTERP_CUBIC) \
                require(PREP_PASTE_UNDER5(warpImageFunc, SRC_PIXEL, DST_PIXEL, INTERP_CUBIC, borderMode)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \
            else if (interpType == INTERP_CUBIC_BSPLINE) \
                require(PREP_PASTE_UNDER5(warpImageFunc, SRC_PIXEL, DST_PIXEL, INTERP_CUBIC_BSPLINE, borderMode)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \
            else \
                REQUIRE(false); \
        }

    if (borderMode == BORDER_ZERO)
        TMP_MACRO(BORDER_ZERO)
    else if (borderMode == BORDER_MIRROR)
        TMP_MACRO(BORDER_MIRROR)
    else
        REQUIRE(false);

    #undef TMP_MACRO

    returnTrue;
}

#endif
