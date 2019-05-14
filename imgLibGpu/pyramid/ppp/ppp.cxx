#if HOSTCODE
#include "ppp.h"
#endif

#include "gpuSupport/gpuTool/gpuToolPlain.h"
#include "pyramid/ppp/pppCommon.h"

#if HOSTCODE
#include "errorLog/errorLog.h"
#include "rndgen/rndgenBase.h"
#include "dataAlloc/arrayMemory.inl"
#include "pyramid/ppp/ppp.h"
#endif

namespace ppp {

//================================================================
//
// getTotalPyramidTileCount
//
//================================================================

#if HOSTCODE

stdbool getTotalPyramidTileCount
(
    const PyramidStructure& pyramid, 
    const Point<Space>& tileSize, 
    Space& resultTileCount, 
    uint32& configHash, 
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    REQUIRE(tileSize >= 1);

    ////

    Space levelCount = pyramid.levelCount();

    RndgenState hash = 0;
    rndgen16(hash); hash ^= levelCount;
    rndgen16(hash); hash ^= tileSize.X;
    rndgen16(hash); hash ^= tileSize.Y;

    ////

    Space totalTileCount = 0;

    for (Space l = 0; l < levelCount; ++l)
    {
        Point<Space> size = pyramid.levelSize(l);
        REQUIRE(size >= 0);

        rndgen16(hash); hash ^= size.X;
        rndgen16(hash); hash ^= size.Y;

        Point<Space> tileCount = divUpNonneg(size, tileSize);
        REQUIRE(tileCount <= maxTileCount());

        totalTileCount += areaOf(tileCount);
    }

    ////

    resultTileCount = totalTileCount;
    configHash = hash; 

    stdEnd;
}

#endif

//================================================================
//
// getTotalPyramidTileCount
//
//================================================================

#if HOSTCODE

stdbool getTotalPyramidTileCount
(
    const GpuPyramidLayout& layout, 
    const Point<Space>& tileSize, 
    Space& resultTotalTileCount, 
    uint32& configHash, 
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    REQUIRE(tileSize >= 1);

    RndgenState hash = 0;

    ////

    Space levelCount = layout.levelCount;
  
    rndgen16(hash); hash ^= levelCount;
    rndgen16(hash); hash ^= tileSize.X;
    rndgen16(hash); hash ^= tileSize.Y;

    Space totalTileCount = 0;

    for (Space l = 0; l < levelCount; ++l)
    {
        Point<Space> size = layout.levels[l].size;
        REQUIRE(size >= 0);

        rndgen16(hash); hash ^= size.X;
        rndgen16(hash); hash ^= size.Y;

        Point<Space> tileCount = divUpNonneg(size, tileSize);
        REQUIRE(tileCount <= maxTileCount());

        totalTileCount += areaOf(tileCount);
    }

    ////

    resultTotalTileCount = totalTileCount;
    configHash = hash; 

    stdEnd;
}

#endif

//================================================================
//
// maxSupportedLevels
//
//================================================================

const Space maxSupportedLevels = 32;

//================================================================
//
// StaticPyramidStructure
//
//================================================================

struct StaticPyramidStructure
{
    Point<Space> tileCount[maxSupportedLevels];
    Space levelCount;
};

//================================================================
//
// computeGuidingArray
//
//================================================================

GPUTOOL_PLAIN_BEG
(
    computeGuidingArray,
    256, false,
    PREP_EMPTY,
    ((StaticPyramidStructure, pyramidStructure))
    ((GpuArray<GuidingElement>, result))
)
#if DEVCODE
{

    ARRAY_EXPOSE(result);
    devDebugCheck(SpaceU(plainGlobalIdx) < SpaceU(resultSize));

    ////

    Space tileLinearIndex = plainGlobalIdx;

    Space remainingTileCount = tileLinearIndex;

    ////

    Space level = 0;
    Point<Space> tileCount = point(0);

    devUnrollLoop

    for (; level < maxSupportedLevels; ++level)
    {
        if_not (level < pyramidStructure.levelCount)
            break;

        tileCount = pyramidStructure.tileCount[level];
        Space levelAreaInTiles = areaOf(tileCount);

        if (remainingTileCount < levelAreaInTiles)
            break;

        remainingTileCount -= levelAreaInTiles;
    }

    ////

    devAbortCheck(SpaceU(remainingTileCount) < SpaceU(areaOf(tileCount)));

    ////

    Space tileY = remainingTileCount / tileCount.X;
    Space tileX = remainingTileCount - tileY * tileCount.X;

    Point<Space> tileIndex = point(tileX, tileY);
    devAbortCheck(tileIndex >= 0 && tileIndex < tileCount);

    ////

    GuidingElement value; 

    devAbortCheck(SpaceU(level) < SpaceU(pyramidStructure.levelCount));
    value.data.level = level;

    devAbortCheck(convertExact<SpaceU>(tileIndex) <= SpaceU(0xFFFF));
    value.data.tileIndex = convertExact<uint16>(tileIndex);

    ////

    resultPtr[plainGlobalIdx] = value;

}
#endif
GPUTOOL_PLAIN_END

//================================================================
//
// prepareGuidingArray
//
//================================================================

#if HOSTCODE

stdbool prepareGuidingArray
(
    const PyramidStructure& pyramid, 
    const Point<Space>& tileSize, 
    const GpuArray<GuidingElement>& result, 
    stdPars(GpuProcessKit)
)
{
    stdBegin;

    Space levelCount = pyramid.levelCount();
    REQUIRE(levelCount <= maxSupportedLevels);
    REQUIRE(tileSize >= 1);

    ////

    StaticPyramidStructure pyramidStructure;
    pyramidStructure.levelCount = levelCount;

    Space totalTileCount = 0;

    for (Space l = 0; l < levelCount; ++l)
    {
        Point<Space> size = pyramid.levelSize(l);
        Point<Space> tileCount = divUpNonneg(size, tileSize);
        REQUIRE(tileCount <= maxTileCount());

        pyramidStructure.tileCount[l] = tileCount;
        totalTileCount += areaOf(tileCount);
    }
    
    ////

    REQUIRE(result.size() == totalTileCount);

    ////

    require(computeGuidingArray(totalTileCount, pyramidStructure, result, stdPass));

    ////

    stdEnd;
}

#endif

//================================================================
//
// checkPyramidGuide
//
//================================================================

#if HOSTCODE

stdbool checkPyramidGuide(const GpuPyramidLayout& layout, const PyramidGuide& guide, stdPars(ErrorLogKit))
{
    stdBegin;

    Space totalTileCount = 0;
    uint32 configHash = 0;
    REQUIRE(guide.tileSize >= 1);
    require(getTotalPyramidTileCount(layout, guide.tileSize, totalTileCount, configHash, stdPass));

    REQUIRE(guide.configHash == configHash);
    REQUIRE(guide.guideArray.size() == totalTileCount);
    REQUIRE(guide.levelCount == layout.levelCount);

    stdEnd;
}

#endif

//================================================================
//
// PyramidGuideMemory::realloc
//
//================================================================

#if HOSTCODE

stdbool PyramidGuideMemory::realloc(const PyramidStructure& pyramid, const Point<Space>& tileSize, stdPars(GpuProcessKit))
{
    stdBegin;

    dealloc();

    ////

    Space totalTileCount = 0;
    uint32 configHash = 0;
    require(getTotalPyramidTileCount(pyramid, tileSize, totalTileCount, configHash, stdPass));

    ////

    require(theGuideArray.realloc(totalTileCount, stdPass));

    ////

    require(prepareGuidingArray(pyramid, tileSize, theGuideArray, stdPass));

    ////

    theLevelCount = pyramid.levelCount();
    theTileSize = tileSize;
    theConfigHash = configHash;

    ////

    stdEnd;
}

#endif

//----------------------------------------------------------------

}
