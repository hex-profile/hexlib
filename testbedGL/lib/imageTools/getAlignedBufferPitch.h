#pragma once

#include "data/space.h"
#include "data/spacex.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// getAlignedBufferPitch
//
//================================================================

inline void getAlignedBufferPitch(Space elemSize, Space sizeX, Space rowByteAlignment, Space& pitch, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0);

    //
    // compute alignment in elements
    //

    REQUIRE(rowByteAlignment >= 1);

    Space alignment = 1;

    if (rowByteAlignment != 1)
    {
        alignment = SpaceU(rowByteAlignment) / SpaceU(elemSize);
        REQUIRE(alignment * elemSize == rowByteAlignment); // divides evenly
    }

    //
    // check the alignment is power of 2
    //

    REQUIRE(isPower2(alignment));
    Space alignmentMask = alignment - 1;

    //
    // align image size X
    //

    Space sizeXplusMask = 0;
    REQUIRE(safeAdd(sizeX, alignmentMask, sizeXplusMask));

    Space alignedSizeX = sizeXplusMask & (~alignmentMask);
    REQUIRE(alignedSizeX >= sizeX); // self-check

    ////

    pitch = alignedSizeX;
}
