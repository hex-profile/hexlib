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
// warpImageFunc
//
//================================================================

#define TMP_MACRO(interpMode, borderMode, SrcPixel, texInterpolation, texStatement) \
    \
    GPUTOOL_2D \
    ( \
        PREP_PASTE_UNDER4(warpImageFunc, PIXEL, interpMode, borderMode), \
        ((const SrcPixel, src, texInterpolation, borderMode)) \
        ((const float32_x2, map, INTERP_LINEAR, BORDER_CLAMP)), \
        ((PIXEL, dst)), \
        ((LinearTransform<Point<float32>>, srcTransform)) \
        ((Point<float32>, mapScaleFactor)) \
        ((Point<float32>, mapValueFactor)), \
        \
        { \
            Point<float32> pos = point(Xs, Ys); \
            float32_x2 offset = tex2D(mapSampler, pos * mapScaleFactor * mapTexstep); \
            Point<float32> srcPos = srcTransform(pos + mapValueFactor * point(offset.x, offset.y)); \
            storeNorm(dst, texStatement); \
        } \
    )

#define TMP_MACRO2(borderMode) \
    TMP_MACRO(INTERP_LINEAR, borderMode, PIXEL, INTERP_LINEAR, tex2D(srcSampler, srcPos * srcTexstep)) \
    TMP_MACRO(INTERP_CUBIC, borderMode, PIXEL, INTERP_NONE, tex2DCubic(srcSampler, srcPos, srcTexstep)) \
    TMP_MACRO(INTERP_CUBIC_BSPLINE, borderMode, typename BsplineExtendedType<PIXEL>::T, INTERP_NONE, tex2DCubicBspline(srcSampler, srcPos, srcTexstep)) \

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
    //----------------------------------------------------------------
    //
    // Direct interpolation modes.
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(borderMode) \
        { \
            if (interpType == INTERP_LINEAR) \
                require(PREP_PASTE_UNDER4(warpImageFunc, PIXEL, INTERP_LINEAR, borderMode)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \
            else if (interpType == INTERP_CUBIC) \
                require(PREP_PASTE_UNDER4(warpImageFunc, PIXEL, INTERP_CUBIC, borderMode)(src, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \
            else if (interpType == INTERP_CUBIC_BSPLINE) \
                ; \
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

        ////

        #define TMP_MACRO(borderMode) \
            require(PREP_PASTE_UNDER4(warpImageFunc, PIXEL, INTERP_CUBIC_BSPLINE, borderMode)(prefilteredImage, map, dst, srcTransform, mapScaleFactor, mapValueFactor, stdPass)); \

        if (borderMode == BORDER_ZERO)
            TMP_MACRO(BORDER_ZERO)
        else if (borderMode == BORDER_MIRROR)
            TMP_MACRO(BORDER_MIRROR)
        else
            REQUIRE(false);

        #undef TMP_MACRO
    }

    ////

    returnTrue;
}

#endif
