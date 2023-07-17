#if HOSTCODE
#include "drawTestImage.h"
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
    ((int32, stripePeriodBits))
    ((int32, stripeWidth))
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

    if (false)
    {
        int32 periodDiv = pos.X >> stripePeriodBits;
        int32 periodRem = pos.Y - (periodDiv << stripePeriodBits);

        if (periodRem < stripeWidth)
        {
            int32 colorType = periodDiv & 3;

            ////

            uint32 colorFore = 0x00000000;
            if (colorType == 0) colorFore = 0x00FFFFFF;
            if (colorType == 1) colorFore = 0x00FF0000;
            if (colorType == 2) colorFore = 0x0000FF00;
            if (colorType == 3) colorFore = 0x000000FF;

            ////

            result = colorFore;
        }
    }

    ////

    *dstImage = recastEqualLayout<uint8_x4>(result);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// drawTestImage
//
//================================================================

#if HOSTCODE

stdbool drawTestImage
(
    const Point<Space>& scrollOfs,
    int32 stripePeriodBits,
    int32 stripeWidth,
    const GpuMatrix<uint8_x4>& dst,
    stdPars(GpuProcessKit)
)
{
    return genTestImage(dst, scrollOfs, stripePeriodBits, stripeWidth, stdPassThru);
}

#endif

