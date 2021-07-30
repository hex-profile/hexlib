#pragma once

//================================================================
//
// GPT_GET_FLAT_PROCESSOR
//
//================================================================

#define GPT_GET_FLAT_PROCESSOR \
    \
    COMPILE_ASSERT(vCellSizeX == 1 && vCellSizeY == 1); \
    \
    Space threadIndex = vTileMemberX + vTileMemberY * vTileSizeX; \
    static constexpr Space threadCount = vTileSizeX * vTileSizeY;
    