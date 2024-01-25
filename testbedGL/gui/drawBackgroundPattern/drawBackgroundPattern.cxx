#if HOSTCODE
#include "drawBackgroundPattern.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "rndgen/rndgenBase.h"

//================================================================
//
// genTestImage
//
//================================================================

GPUTOOL_2D_BEG
(
    genTestImage,
    PREP_EMPTY,
    ((uint8_x4, dstImage)),
    ((Point<Space>, scrollOfs))
)
#if DEVCODE
{
    Point<Space> pos = point(X, Y) + scrollOfs;

    ////

    uint32 r = distributiveHash(0x727ABEFA + pos.X + pos.Y * 0xEAE749B3);

    uint32 shift = (r % 4);

    uint32 colorBack = distributiveHash(0x0B7B64C0 + (pos.X >> shift) + (pos.Y >> shift) * 0x01090F4D);

    {
        uint32 v = colorBack & 0xFF;
        v = 0x80 + ((int32(v) - 0x80) >> 1);
        colorBack = v + (v << 8) + (v << 16);
    }

    ////

    auto result = colorBack;

    *dstImage = recastEqualLayout<uint8_x4>(result);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// drawBackgroundPattern
//
//================================================================

#if HOSTCODE

stdbool drawBackgroundPattern
(
    const Point<Space>& scrollOfs,
    const GpuMatrix<uint8_x4>& dst,
    stdPars(GpuProcessKit)
)
{
    return genTestImage(dst, scrollOfs, stdPassThru);
}

#endif

