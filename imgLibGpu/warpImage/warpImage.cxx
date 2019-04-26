#include "warpImage.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuDevice/loadstore/storeNorm.h"

//================================================================
//
// warpImageFunc
//
//================================================================

using Pixel = uint8_x4;

//----------------------------------------------------------------

GPUTOOL_2D_BEG
(
    warpImageFunc,
    ((const Pixel, src, INTERP_LINEAR, BORDER_ZERO))
    ((const float32_x2, map, INTERP_LINEAR, BORDER_CLAMP)),
    ((Pixel, dst)),
    ((Point<float32>, mapScaleFactor))
    ((Point<float32>, mapValueFactor))
)
#if DEVCODE
{
    Point<float32> pos = point(Xs, Ys);
    float32_x2 offset = tex2D(mapSampler, pos * mapScaleFactor * mapTexstep);
    auto ofs = mapValueFactor * point(offset.x, offset.y);

    auto value = tex2D(srcSampler, (pos + ofs) * srcTexstep);
    storeNorm(dst, value);
}
#endif
GPUTOOL_2D_END

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
    REQUIRE(interpType == INTERP_LINEAR);

    require(warpImageFunc(src, map, dst, mapScaleFactor, mapValueFactor, stdPass));

    stdEnd;
}

#endif
