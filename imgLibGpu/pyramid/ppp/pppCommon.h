#pragma once

#include "data/gpuArray.h"
#include "point/point.h"

namespace ppp {

//================================================================
//
// GuidingElement
//
//================================================================

sysinline Point<Space> maxTileCount() {return point(Space(1) << 16);}

//----------------------------------------------------------------

struct GuidingElementData
{
    Point<uint16> tileIndex;
    uint32 level;
};

union GuidingElement
{
    GuidingElementData data;
    uint64 asInt;
};

COMPILE_ASSERT(sizeof(GuidingElement) == 8);

//----------------------------------------------------------------

sysinline Point<Space> getGlobalIdx(const GuidingElement& guide, const Point<Space>& tileSize, const Point<Space>& tileMember)
{
    return convertExact<Space>(guide.data.tileIndex) * tileSize + tileMember;
}

//================================================================
//
// PyramidGuide
//
//================================================================

struct PyramidGuide
{
    GpuArray<GuidingElement> guideArray;
    Space levels;
    Point<Space> tileSize;
    uint32 configHash;
};

//----------------------------------------------------------------

sysinline PyramidGuide pyramidGuide(const GpuArray<GuidingElement>& guideArray, Space levels, const Point<Space>& tileSize, uint32 configHash)
{
    PyramidGuide result;
    result.guideArray = guideArray;
    result.levels = levels;
    result.tileSize = tileSize;
    result.configHash = configHash;
    return result;
}

//================================================================
//
// pppDefaultTile
//
//================================================================

#define PPP_DEFAULT_TILE_SIZE_X 32
#define PPP_DEFAULT_TILE_SIZE_Y 8

sysinline Point<Space> defaultTileSize() {return point(PPP_DEFAULT_TILE_SIZE_X, PPP_DEFAULT_TILE_SIZE_Y);}

//----------------------------------------------------------------

}
