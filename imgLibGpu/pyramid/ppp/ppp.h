#pragma once

#include "pyramid/gpuPyramid.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuProcessHeader.h"
#include "pppCommon.h"

namespace ppp {

//================================================================
//
// getTotalPyramidTileCount
//
// Computes area (as for single-layered pyramid).
// The area is measured in tiles.
//
//================================================================

stdbool getTotalPyramidTileCount
(
    const PyramidStructure& pyramid,
    const Point<Space>& tileSize,
    Space& resultTileCount,
    uint32& configHash,
    stdPars(ErrorLogKit)
);

//================================================================
//
// prepareGuidingArray
//
//================================================================

stdbool prepareGuidingArray
(
    const PyramidStructure& pyramid,
    const Point<Space>& tileSize,
    const GpuArray<GuidingElement>& result,
    stdPars(GpuProcessKit)
);

//----------------------------------------------------------------

stdbool checkPyramidGuide(const GpuPyramidLayout& layout, const PyramidGuide& guide, stdPars(ErrorLogKit));

//================================================================
//
// PyramidGuideMemory
//
//================================================================

class PyramidGuideMemory
{

public:

    sysinline operator PyramidGuide () const
        {return pyramidGuide(theGuideArray, theLevelCount, theTileSize, theConfigHash);}

    sysinline GpuArray<GuidingElement> guideArray() const
        {return theGuideArray;}

public:

    void dealloc()
    {
        theGuideArray.dealloc();
        theConfigHash = 0;
        theTileSize = point(0);
    }

    stdbool realloc(const PyramidStructure& pyramid, const Point<Space>& tileSize, stdPars(GpuProcessKit));

public:

    sysinline Space getTotalPixelArea() const
    {
        return theGuideArray.size() * areaOf(theTileSize);
    }

private:

    GpuArrayMemory<GuidingElement> theGuideArray;
    Space theLevelCount = 0;
    Point<Space> theTileSize = point(0);
    uint32 theConfigHash = 0;

};

//----------------------------------------------------------------

}
