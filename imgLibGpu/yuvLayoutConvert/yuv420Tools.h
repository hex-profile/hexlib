#pragma once

#include "data/spacex.h"

//================================================================
//
// yuv420SizeValid
// yuv420TotalArea
//
//================================================================

inline bool yuv420SizeValid(const Point<Space>& frameSize)
{
    require(frameSize >= 0);
    require((frameSize & 1) == 0); // even dimensions

    Space frameArea = 0;
    require(safeMul(frameSize.X, frameSize.Y, frameArea));

    Space frameBytes = 0;
    require(safeAdd(frameArea, frameArea >> 1, frameBytes));

    return true;
}

//----------------------------------------------------------------

inline Space yuv420TotalArea(const Point<Space>& frameSize)
{
    Space area = frameSize.X * frameSize.Y;
    return area + (area >> 1);
}
