#pragma once

//================================================================
//
// GPT_GET_FLAT_PROCESSOR_2D
//
//================================================================

#define GPT_GET_FLAT_PROCESSOR_2D \
    \
    COMPILE_ASSERT(vCellSizeX == 1 && vCellSizeY == 1); \
    \
    constexpr Space threadCount = vTileSizeX * vTileSizeY; \
    MAKE_VARIABLE_USED(threadCount); \
    \
    const Space threadIndex = vTileMemberX + vTileMemberY * vTileSizeX; \
    MAKE_VARIABLE_USED(threadIndex); \
    \
    const bool threadIsMain = (threadIndex == 0); \
    MAKE_VARIABLE_USED(threadIsMain)

//================================================================
//
// GPT_GET_FLAT_PROCESSOR_1D
//
//================================================================

#define GPT_GET_FLAT_PROCESSOR_1D \
    \
    constexpr Space threadCount = vTileSize; \
    MAKE_VARIABLE_USED(threadCount); \
    \
    const Space threadIndex = vTileMember; \
    MAKE_VARIABLE_USED(threadIndex); \
    \
    const bool threadIsMain = (threadIndex == 0); \
    MAKE_VARIABLE_USED(threadIsMain)
