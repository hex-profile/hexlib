#pragma once

#include "numbers/float/floatType.h"
#include "data/space.h"

//================================================================
//
// convertToNearestIndex
//
//================================================================

template <typename Type>
sysinline auto convertToNearestIndex(const Type& pos)
    {return convertDown<Space>(pos);} // equivalent to convertNearest(pos - 0.5f)

//================================================================
//
// convertIndexToPos
//
//================================================================

template <typename Type>
sysinline auto convertIndexToPos(const Type& idx)
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

template <typename Type>
sysinline Type roundPosToNearestSample(const Type& pos)
    {return floorv(pos) + 0.5f;}

//================================================================
//
// computeFilterStartIndex
// computeFilterStartPos
//
// Center position is specified in SPACE format.
//
//================================================================

template <typename FilterCenter, typename Taps>
sysinline auto computeFilterStartIndex(const FilterCenter& filterCenter, const Taps& taps)
{
    auto startPos = filterCenter - 0.5f * convertFloat32(taps - 1); // taps-1 is a must!
    return convertToNearestIndex(startPos);
}

//----------------------------------------------------------------

template <typename FilterCenter, typename Taps>
sysinline auto computeFilterStartPos(const FilterCenter& filterCenter, const Taps& taps)
{
    return convertIndexToPos(computeFilterStartIndex(filterCenter, taps));
}
