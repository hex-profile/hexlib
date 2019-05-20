#include "warpImage.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"
#include "bsplinePrefilter/bsplinePrefilterSettings.h"
#include "bsplinePrefilter/bsplinePrefilter.h"
#include "numbers/lt/ltType.h"

#if HOSTCODE
#include "dataAlloc/gpuMatrixMemory.h"
#endif

//================================================================
//
// warpImageFunc
//
//================================================================

#define TMP_MACRO(interpMode, SrcPixel, texInterpolation, texStatement) \
    \
    GPUTOOL_2D \
    ( \
        PREP_PASTE4(warpImageFunc_, PIXEL, _, interpMode), \
        ((const SrcPixel, src, texInterpolation, BORDER_ZERO)) \
        ((const float32_x2, map, INTERP_LINEAR, BORDER_CLAMP)), \
        ((PIXEL, dst)), \
        ((LinearTransform<Point<float32>>, srcTransform)) \
        ((Point<float32>, mapScaleFactor)) \
        ((Point<float32>, mapValueFactor)), \
        \
        { \
            Point<float32> pos = point(Xs, Ys); \
            float32_x2 offset = tex2D(mapSampler, pos * mapScaleFactor * mapTexstep); \
            Point<float32> srcPos = ltApply(pos + mapValueFactor * point(offset.x, offset.y), srcTransform); \
            storeNorm(dst, texStatement); \
        } \
    )

TMP_MACRO(INTERP_LINEAR, PIXEL, INTERP_LINEAR, tex2D(srcSampler, srcPos * srcTexstep))
TMP_MACRO(INTERP_CUBIC, PIXEL, INTERP_NONE, texCubic2D(srcSampler, srcPos, srcTexstep))
TMP_MACRO(INTERP_CUBIC_BSPLINE, typename BsplineExtendedType<PIXEL>::T, INTERP_NONE, texCubicBspline2D(srcSampler, srcPos, srcTexstep))

#undef TMP_MACRO

//================================================================
//
// warpImage
//
//================================================================

#if HOSTCODE

template <>
stdbool warpImage
(
    const GpuMatrix<const PIXEL>& src,
    const LinearTransform<Point<float32>>& srcTransform,
    const GpuMatrix<const float32_x2>& map,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    BorderMode borderMode,
    InterpType interpType,
    const GpuMatrix<PIXEL> dst,
    stdPars(GpuProcessKit)
)
{
    stdBegin;

    REQUIRE(borderMode == BORDER_ZERO);

    //----------------------------------------------------------------
    //
    // Direct interpolation modes.
    //
    //----------------------------------------------------------------

    if (interpType == INTERP_LINEAR)
        require(PREP_PASTE3(warpImageFunc_, PIXEL, _INTERP_LINEAR)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass));
    else if (interpType == INTERP_CUBIC)
        require(PREP_PASTE3(warpImageFunc_, PIXEL, _INTERP_CUBIC)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass));
    else if (interpType == INTERP_CUBIC_BSPLINE)
        ;
    else
        REQUIRE(false);

    //----------------------------------------------------------------
    //
    // Special mode with prefiltering.
    //
    //----------------------------------------------------------------

    if (interpType == INTERP_CUBIC_BSPLINE)
    {
        using IntermType = typename BsplineExtendedType<PIXEL>::T;

        GPU_MATRIX_ALLOC(prefilteredImage, IntermType, src.size());
        require((bsplineCubicPrefilter<PIXEL, IntermType, IntermType>(src, prefilteredImage, point(1.f), BORDER_MIRROR, stdPass)));

        require(PREP_PASTE3(warpImageFunc_, PIXEL, _INTERP_CUBIC_BSPLINE)(prefilteredImage, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass));
    }

    ////

    stdEnd;
}

#endif
