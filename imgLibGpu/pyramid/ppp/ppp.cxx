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

void getTotalPyramidTileCount
(
    const PyramidStructure& pyramid,
    const Point<Space>& tileSize,
    Space& resultTileCount,
    uint32& configHash,
    stdPars(ErrorLogKit)
)
{
    REQUIRE(tileSize >= 1);

    ////

    Space levels = pyramid.levels();

    RndgenState hash = 0;
    rndgen16(hash); hash ^= levels;
    rndgen16(hash); hash ^= tileSize.X;
    rndgen16(hash); hash ^= tileSize.Y;

    ////

    Space totalTileCount = 0;

    for_count (l, levels)
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
}

#endif

//================================================================
//
// getTotalPyramidTileCount
//
//================================================================

#if HOSTCODE

void getTotalPyramidTileCount
(
    const GpuPyramidLayout& layout,
    const Point<Space>& tileSize,
    Space& resultTotalTileCount,
    uint32& configHash,
    stdPars(ErrorLogKit)
)
{
    REQUIRE(tileSize >= 1);

    RndgenState hash = 0;

    ////

    Space levels = layout.levels;

    rndgen16(hash); hash ^= levels;
    rndgen16(hash); hash ^= tileSize.X;
    rndgen16(hash); hash ^= tileSize.Y;

    Space totalTileCount = 0;

    for_count (l, levels)
    {
        Point<Space> size = layout.levelData[l].size;
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
    Space levels;
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
        if_not (level < pyramidStructure.levels)
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

    devAbortCheck(SpaceU(level) < SpaceU(pyramidStructure.levels));
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

void prepareGuidingArray
(
    const PyramidStructure& pyramid,
    const Point<Space>& tileSize,
    const GpuArray<GuidingElement>& result,
    stdPars(GpuProcessKit)
)
{
    Space levels = pyramid.levels();
    REQUIRE(levels <= maxSupportedLevels);
    REQUIRE(tileSize >= 1);

    ////

    StaticPyramidStructure pyramidStructure;
    pyramidStructure.levels = levels;

    Space totalTileCount = 0;

    for_count (l, levels)
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

    computeGuidingArray(totalTileCount, pyramidStructure, result, stdPass);
}

#endif

//================================================================
//
// checkPyramidGuide
//
//================================================================

#if HOSTCODE

void checkPyramidGuide(const GpuPyramidLayout& layout, const PyramidGuide& guide, stdPars(ErrorLogKit))
{
    Space totalTileCount = 0;
    uint32 configHash = 0;
    REQUIRE(guide.tileSize >= 1);
    getTotalPyramidTileCount(layout, guide.tileSize, totalTileCount, configHash, stdPass);

    REQUIRE(guide.configHash == configHash);
    REQUIRE(guide.guideArray.size() == totalTileCount);
    REQUIRE(guide.levels == layout.levels);
}

#endif

//================================================================
//
// PyramidGuideMemory::realloc
//
//================================================================

#if HOSTCODE

void PyramidGuideMemory::realloc(const PyramidStructure& pyramid, const Point<Space>& tileSize, stdPars(GpuProcessKit))
{
    dealloc();

    ////

    Space totalTileCount = 0;
    uint32 configHash = 0;
    getTotalPyramidTileCount(pyramid, tileSize, totalTileCount, configHash, stdPass);

    ////

    theGuideArray.realloc(totalTileCount, stdPass);

    ////

    prepareGuidingArray(pyramid, tileSize, theGuideArray, stdPass);

    ////

    theLevels = pyramid.levels();
    theTileSize = tileSize;
    theConfigHash = configHash;
}

#endif

//----------------------------------------------------------------

}
