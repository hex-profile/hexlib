#pragma once

#include "numbers/float/floatType.h"
#include "point/point.h"
#include "data/space.h"

//================================================================
//
// convertToNearestIndex
//
//================================================================

sysinline Point<Space> convertToNearestIndex(const Point<float32>& pos)
    {return convertDown<Space>(pos);} // equivalent to convertNearest(pos - 0.5f)

sysinline Space convertToNearestIndex(float32 pos)
    {return convertDown<Space>(pos);}

//================================================================
//
// convertIndexToPos
//
//================================================================

sysinline Point<float32> convertIndexToPos(const Point<Space>& idx)
    {return convertFloat32(idx) + 0.5f;}

sysinline float32 convertIndexToPos(Space idx)
    {return convertFloat32(idx) + 0.5f;}

//================================================================
//
// convertPosToIndex
//
//================================================================

template <typename Type>
sysinline auto convertPosToIndex(const Type& pos)
    {return pos - 0.5f;}

//================================================================
//
// roundPosToNearestSample
//
// result = indexToPos(round(posToIndex(pos)))
// result = round(pos - 0.5) + 0.5
// result = floor(pos - 0.5 + 0.5) + 0.5
// result = floor(pos) + 0.5
//
//================================================================

sysinline float32 roundPosToNearestSample(float32 pos)
    {return floorf(pos) + 0.5f;}

sysinline Point<float32> roundPosToNearestSample(const Point<float32>& pos)
    {return point(roundPosToNearestSample(pos.X), roundPosToNearestSample(pos.Y));}

//================================================================
//
// computeFilterStartIndex
// computeFilterStartPos
//
// Center position is specified in SPACE format.
//
//================================================================

sysinline Point<Space> computeFilterStartIndex(const Point<float32>& filterCenter, const Point<Space>& taps)
{
    Point<float32> startPos = filterCenter - 0.5f * convertFloat32(taps - 1); // taps-1 is a must!
    return convertToNearestIndex(startPos);
}

//----------------------------------------------------------------

sysinline Point<float32> computeFilterStartPos(const Point<float32>& filterCenter, const Point<Space>& taps)
{
    return convertIndexToPos(computeFilterStartIndex(filterCenter, taps));
}

//----------------------------------------------------------------

sysinline Point<Space> computeFilterStartIndex(const Point<float32>& filterCenter, Space taps)
    {return computeFilterStartIndex(filterCenter, point(taps));}

sysinline Point<float32> computeFilterStartPos(const Point<float32>& filterCenter, Space taps)
    {return computeFilterStartPos(filterCenter, point(taps));}
