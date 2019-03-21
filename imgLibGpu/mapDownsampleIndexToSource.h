#pragma once

#include "data/space.h"

//================================================================
//
// mapDownsampleIndexToSource
//
//================================================================

template <Space downsampleFactor, Space filterSize>
sysinline Space mapDownsampleIndexToSource(Space dstIdx)
{
    //
    // radius = 0.5 * (filterSize - 1)
    //

    const Space radius2 = (filterSize - 1); // compile-time

    //
    // dstPos = dstIdx + 0.5, to space coords
    //
    // dstPos2 = 2*dstIdx + 1;
    // srcPos2 = downsampleFactor * dstPos2 - radius2;
    //
    // Equivalent to:
    // Space srcPos2 = (downsampleFactor * 2) * dstIdx + (downsampleFactor - radius2);
    //

    //
    // To nearest grid, equivalent to convertNearest(srcPos - 0.5):
    // srcIdx = srcPos2 >> 1
    //

    return // one IMAD instruction
        downsampleFactor * dstIdx +
        ((downsampleFactor - radius2) >> 1);
}
