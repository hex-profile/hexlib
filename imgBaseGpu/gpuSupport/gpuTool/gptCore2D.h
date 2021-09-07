#pragma once

#include "gptCommonImpl.h"

//================================================================
//
// GPT_MAKE_KERNEL_2D_*
//
//================================================================

#define GPT_MAKE_KERNEL_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeXp, tileSizeYp, cellSizeXp, cellSizeYp, superTaskSupport, keepAllThreads) \
    \
    devDefineKernel(prefix##Kernel, prefix##Params, o) \
    { \
        constexpr Space vTileSizeX = (tileSizeXp); \
        constexpr Space vTileSizeY = (tileSizeYp); \
        COMPILE_ASSERT(vTileSizeX >= 1 && vTileSizeY >= 1); \
        const auto vTileSize = point(vTileSizeX, vTileSizeY); \
        \
        constexpr Space vCellSizeX = (cellSizeXp); \
        constexpr Space vCellSizeY = (cellSizeYp); \
        COMPILE_ASSERT(vCellSizeX >= 1 && vCellSizeY >= 1); \
        \
        const Space vTileMemberX = (vCellSizeX == 1) ? devThreadX : SpaceU(devThreadX) / SpaceU(vCellSizeX); \
        const Space vTileMemberY = (vCellSizeY == 1) ? devThreadY : SpaceU(devThreadY) / SpaceU(vCellSizeY); \
        const auto vTileMember = point(vTileMemberX, vTileMemberY); MAKE_VARIABLE_USED(vTileMember); \
        \
        const Space vCellMemberX = (vCellSizeX == 1) ? 0 : SpaceU(devThreadX) % SpaceU(vCellSizeX); MAKE_VARIABLE_USED(vCellMemberX); \
        const Space vCellMemberY = (vCellSizeY == 1) ? 0 : SpaceU(devThreadY) % SpaceU(vCellSizeY); MAKE_VARIABLE_USED(vCellMemberY); \
        const auto vCellMember = point(vCellMemberX, vCellMemberY); MAKE_VARIABLE_USED(vCellMember); \
        \
        const Point<Space> vTileIdx = devGroupIdx; \
        const Point<Space> vTileOrg = vTileIdx * vTileSize; \
        \
        const Space X = (vTileSizeX == 1) ? devGroupX : (vTileOrg.X + vTileMember.X); \
        const Space Y = (vTileSizeY == 1) ? devGroupY : (vTileOrg.Y + vTileMember.Y); \
        \
        PREP_IF(superTaskSupport, const Space vSuperTaskCount = devGroupCountZ; MAKE_VARIABLE_USED(vSuperTaskCount);) \
        PREP_IF(superTaskSupport, const Space vSuperTaskIdx = devGroupZ; MAKE_VARIABLE_USED(vSuperTaskIdx);) \
        \
        const float32 Xs = X + 0.5f; MAKE_VARIABLE_USED(Xs); \
        const float32 Ys = Y + 0.5f; MAKE_VARIABLE_USED(Ys); \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_EXPOSE_SAMPLER, prefix) \
        GPT_FOREACH(matrixList, GPT_EXPOSE_MATRIX) \
        GPT_FOREACH(paramList, GPT_EXPOSE_PARAM) \
        \
        const auto& vGlobSize = o.globSize; \
        \
        const bool vItemIsActive = \
            ((vTileSizeX == 1) || (SpaceU(X) < SpaceU(vGlobSize.X))) && \
            ((vTileSizeY == 1) || (SpaceU(Y) < SpaceU(vGlobSize.Y))); \
        \
        if_not (keepAllThreads) \
            if_not (vItemIsActive) \
                return; \
        \
        {

#define GPT_MAKE_KERNEL_2D_END \
        } \
    }

//================================================================
//
// GPT_MAKE_CALLER_2D
//
//================================================================

#define GPT_MAKE_CALLER_2D(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskSupport) \
    \
    hostDeclareKernel(prefix##Kernel, prefix##Params, o); \
    \
    stdbool prefix \
    ( \
        PREP_IF(superTaskSupport, Space superTaskCount) PREP_IF_COMMA(superTaskSupport) \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(matrixList, GPT_DECLARE_MATRIX_ARG) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \
    { \
        if_not (kit.dataProcessing) \
            returnTrue; \
        \
        auto globSize = point<Space>(0); \
        GPT_FOREACH(matrixList, GPT_GET_SIZE) \
        \
        GPT_FOREACH(matrixList, GPT_CHECK_SIZE) \
        \
        PREP_IF(PREP_NOT(superTaskSupport), constexpr Space superTaskCount = 1;) \
        \
        Space groupCountX = divUpNonneg(globSize.X, tileSizeX); \
        Space groupCountY = divUpNonneg(globSize.Y, tileSizeY); \
        REQUIRE(superTaskCount >= 1); \
        \
        if_not (groupCountX && groupCountY && superTaskCount) \
            returnTrue; \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_BIND_SAMPLER, prefix) \
        \
        prefix##Params kernelParams; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_SET_SAMPLER_FIELD, o) \
        GPT_FOREACH(matrixList, GPT_SET_MATRIX_FIELD) \
        kernelParams.globSize = globSize; \
        GPT_FOREACH(paramList, GPT_SET_PARAM_FIELD) \
        \
        require \
        ( \
            kit.gpuKernelCalling.callKernel \
            ( \
                point3D(groupCountX, groupCountY, superTaskCount), \
                point((tileSizeX) * (cellSizeX), (tileSizeY) * (cellSizeY)), \
                areaOf(globSize), \
                prefix##Kernel, \
                kernelParams, \
                kit.gpuCurrentStream, \
                stdPassLocationMsg("Kernel") \
            ) \
        ); \
        \
        returnTrue; \
    }

//================================================================
//
// GPT_CALLER_2D_PROTO
//
//================================================================

#define GPT_CALLER_2D_PROTO(prefix, samplerList, matrixList, paramList) \
    \
    stdbool prefix \
    ( \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(matrixList, GPT_DECLARE_MATRIX_ARG) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \

//================================================================
//
// GPT_MAIN_2D_BEG
// GPT_MAIN_2D_END
//
//================================================================

#define GPT_MAIN_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskSupport, keepAllThreads) \
    GPT_DECLARE_PARAMS(prefix, Point<Space>, samplerList, matrixList, paramList) \
    DEV_ONLY(GPT_DEFINE_SAMPLERS(prefix, samplerList)) \
    HOST_ONLY(GPT_MAKE_CALLER_2D(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskSupport)) \
    DEV_ONLY(GPT_MAKE_KERNEL_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskSupport, keepAllThreads))

#define GPT_MAIN_2D_END \
    DEV_ONLY(GPT_MAKE_KERNEL_2D_END)

//================================================================
//
// GPUTOOL_2D_PROTO
//
//================================================================

#define GPUTOOL_2D_PROTO(prefix, samplerSeq, matrixSeq, paramSeq) \
    GPT_CALLER_2D_PROTO(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o))

//================================================================
//
// GPUTOOL_2D_BEG_EX2
// GPUTOOL_2D_END_EX2
//
//================================================================

#define GPUTOOL_2D_BEG_EX2(prefix, tileSize, cellSize, superTaskSupport, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPT_MAIN_2D_BEG(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o), PREP_ARG2_0 tileSize, PREP_ARG2_1 tileSize, \
        PREP_ARG2_0 cellSize, PREP_ARG2_1 cellSize, superTaskSupport, keepAllThreads) \

#define GPUTOOL_2D_END_EX2 \
    GPT_MAIN_2D_END

//================================================================
//
// GPUTOOL_2D_BEG_EX
// GPUTOOL_2D_END_EX
//
//================================================================

#define GPUTOOL_2D_BEG_EX(prefix, tileSize, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPUTOOL_2D_BEG_EX2(prefix, tileSize, (1, 1), PREP_FALSE, keepAllThreads, samplerSeq, matrixSeq, paramSeq)

#define GPUTOOL_2D_END_EX \
    GPUTOOL_2D_END_EX2

//================================================================
//
// GPUTOOL_2D
// GPUTOOL_2D_BEG
// GPUTOOL_2D_END
//
//================================================================

#define GPUTOOL_2D_DEFAULT_THREAD_COUNT (32, 8)

//----------------------------------------------------------------

#define GPUTOOL_2D_BEG(prefix, samplerSeq, matrixSeq, paramSeq) \
    GPUTOOL_2D_BEG_EX(prefix, GPUTOOL_2D_DEFAULT_THREAD_COUNT, false, samplerSeq, matrixSeq, paramSeq)

#define GPUTOOL_2D_END \
    GPUTOOL_2D_END_EX

#define GPUTOOL_2D(prefix, samplerSeq, matrixSeq, paramSeq, iterationBody) \
    GPUTOOL_2D_BEG_EX(prefix, GPUTOOL_2D_DEFAULT_THREAD_COUNT, false, samplerSeq, matrixSeq, paramSeq) \
    DEV_ONLY(iterationBody) \
    GPUTOOL_2D_END_EX
