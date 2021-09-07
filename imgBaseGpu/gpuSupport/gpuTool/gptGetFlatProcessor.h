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
    constexpr Space threadCount = vTileSizeX * vTileSizeY; MAKE_VARIABLE_USED(threadCount); \
    \
    const Space threadIndex = vTileMemberX + vTileMemberY * vTileSizeX; \
    const Space threadIsMain = (threadIndex == 0)
