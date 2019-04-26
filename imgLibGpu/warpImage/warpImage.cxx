#include "warpImage.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"
#include "bsplinePrefilter/bsplinePrefilterSettings.h"
#include "bsplinePrefilter/bsplinePrefilter.h"

#if HOSTCODE
#include "dataAlloc/gpuMatrixMemory.h"
#endif

//================================================================
//
// warpImageFunc
//
//================================================================

using Pixel = uint8_x4;

//----------------------------------------------------------------

#define TMP_MACRO(interpMode, SrcPixel, texInterpolation, texStatement) \
    \
    GPUTOOL_2D \
    ( \
        PREP_PASTE2(warpImageFunc_, interpMode), \
        ((const SrcPixel, src, texInterpolation, BORDER_ZERO)) \
        ((const float32_x2, map, INTERP_LINEAR, BORDER_CLAMP)), \
        ((Pixel, dst)), \
        ((Point<float32>, mapScaleFactor)) \
        ((Point<float32>, mapValueFactor)), \
        \
        { \
            Point<float32> pos = point(Xs, Ys); \
            float32_x2 offset = tex2D(mapSampler, pos * mapScaleFactor * mapTexstep); \
            auto ofs = mapValueFactor * point(offset.x, offset.y); \
            \
            auto value = texStatement; \
            storeNorm(dst, value); \
        } \
    )

TMP_MACRO(INTERP_LINEAR, Pixel, INTERP_LINEAR, tex2D(srcSampler, (pos + ofs) * srcTexstep))
TMP_MACRO(INTERP_CUBIC, Pixel, INTERP_NONE, texCubic2D(srcSampler, pos + ofs, srcTexstep))
TMP_MACRO(INTERP_CUBIC_BSPLINE, typename BsplineExtendedType<Pixel>::T, INTERP_NONE, texCubicBspline2D(srcSampler, pos + ofs, srcTexstep))

#undef TMP_MACRO

//================================================================
//
// warpImage
//
//================================================================

#if HOSTCODE

template <>
bool warpImage
(
    const GpuMatrix<const Pixel>& src,
    const GpuMatrix<const float32_x2>& map,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    BorderMode borderMode,
    InterpType interpType,
    const GpuMatrix<Pixel> dst,
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
        require(warpImageFunc_INTERP_LINEAR(src, map, dst, mapScaleFactor, mapValueFactor, stdPass));
    else if (interpType == INTERP_CUBIC)
        require(warpImageFunc_INTERP_CUBIC(src, map, dst, mapScaleFactor, mapValueFactor, stdPass));
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
        using IntermType = typename BsplineExtendedType<Pixel>::T;

        GPU_MATRIX_ALLOC(prefilteredImage, IntermType, src.size());
        require((bsplineCubicPrefilter<Pixel, IntermType, IntermType>(src, prefilteredImage, point(1.f), BORDER_MIRROR, stdPass)));

        require(warpImageFunc_INTERP_CUBIC_BSPLINE(prefilteredImage, map, dst, mapScaleFactor, mapValueFactor, stdPass));
    }

    ////

    stdEnd;
}

#endif
