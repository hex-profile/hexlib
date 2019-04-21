#pragma once

#include "gptCommonImpl.h"

//================================================================
//
// GPT_MAKE_KERNEL_2D_*
//
//================================================================

#define GPT_MAKE_KERNEL_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeXp, tileSizeYp, cellSizeXp, cellSizeYp, superTaskCount, keepAllThreads) \
    \
    devDefineKernel(prefix##Kernel, prefix##Params, o) \
    { \
        const Space vTileSizeX = (tileSizeXp); \
        const Space vTileSizeY = (tileSizeYp); \
        COMPILE_ASSERT(vTileSizeX >= 1 && vTileSizeY >= 1); \
        Point<Space> vTileSize = point(vTileSizeX, vTileSizeY); \
        \
        const Space vCellSizeX = (cellSizeXp); \
        const Space vCellSizeY = (cellSizeYp); \
        COMPILE_ASSERT(vCellSizeX >= 1 && vCellSizeY >= 1); \
        \
        const Space vTileMemberX = (vCellSizeX == 1) ? devThreadX : SpaceU(devThreadX) / SpaceU(vCellSizeX); \
        const Space vTileMemberY = (vCellSizeY == 1) ? devThreadY : SpaceU(devThreadY) / SpaceU(vCellSizeY); \
        Point<Space> vTileMember = point(vTileMemberX, vTileMemberY); MAKE_VARIABLE_USED(vTileMember); \
        \
        const Space vCellMemberX = (vCellSizeX == 1) ? 0 : SpaceU(devThreadX) % SpaceU(vCellSizeX); MAKE_VARIABLE_USED(vCellMemberX); \
        const Space vCellMemberY = (vCellSizeY == 1) ? 0 : SpaceU(devThreadY) % SpaceU(vCellSizeY); MAKE_VARIABLE_USED(vCellMemberY); \
        Point<Space> vCellMember = point(vCellMemberX, vCellMemberY); MAKE_VARIABLE_USED(vCellMember); \
        \
        Point<Space> vTileIdx = devGroupIdx; \
        Point<Space> vTileOrg = vTileIdx * vTileSize; \
        \
        Space X = (vTileSizeX == 1) ? devGroupX : (vTileOrg.X + vTileMember.X); \
        Space Y = (vTileSizeY == 1) ? devGroupY : (vTileOrg.Y + vTileMember.Y); \
        \
        const Space vSuperTaskCount = (superTaskCount); MAKE_VARIABLE_USED(vSuperTaskCount); \
        Space vSuperTaskIdx = devGroupZ; MAKE_VARIABLE_USED(vSuperTaskIdx); \
        \
        float32 Xs = X + 0.5f; MAKE_VARIABLE_USED(Xs); \
        float32 Ys = Y + 0.5f; MAKE_VARIABLE_USED(Ys); \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_EXPOSE_SAMPLER, prefix) \
        GPT_FOREACH(matrixList, GPT_EXPOSE_MATRIX) \
        GPT_FOREACH(paramList, GPT_EXPOSE_PARAM) \
        \
        const Point<Space>& vGlobSize = o.globSize; \
        \
        bool vItemIsActive = \
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

#define GPT_MAKE_CALLER_2D(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskCount) \
    \
    hostDeclareKernel(prefix##Kernel, prefix##Params, o); \
    \
    stdbool prefix \
    ( \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(matrixList, GPT_DECLARE_MATRIX_ARG) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \
    { \
        stdBegin; \
        \
        if_not (kit.dataProcessing) \
            return true; \
        \
        Point<Space> globSize = point(0); \
        GPT_FOREACH(matrixList, GPT_GET_SIZE) \
        \
        GPT_FOREACH(matrixList, GPT_CHECK_SIZE) \
        \
        Space groupCountX = divUpNonneg(globSize.X, tileSizeX); \
        Space groupCountY = divUpNonneg(globSize.Y, tileSizeY); \
        REQUIRE(superTaskCount >= 0); \
        \
        if_not (groupCountX && groupCountY && superTaskCount) \
            return true; \
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
        stdEnd; \
    }

//================================================================
//
// GPT_CALLER_PROTO_2D
//
//================================================================

#define GPT_CALLER_PROTO_2D(prefix, samplerList, matrixList, paramList) \
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

#define GPT_MAIN_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskCount, keepAllThreads) \
    GPT_DECLARE_PARAMS(prefix, Point<Space>, samplerList, matrixList, paramList) \
    GPT_DEFINE_SAMPLERS(prefix, samplerList) \
    HOST_ONLY(GPT_MAKE_CALLER_2D(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskCount)) \
    DEV_ONLY(GPT_MAKE_KERNEL_2D_BEG(prefix, samplerList, matrixList, paramList, tileSizeX, tileSizeY, cellSizeX, cellSizeY, superTaskCount, keepAllThreads))

#define GPT_MAIN_2D_END \
    DEV_ONLY(GPT_MAKE_KERNEL_2D_END)

//================================================================
//
// GPUTOOL_PROTO_2D
//
//================================================================

#define GPUTOOL_PROTO_2D(prefix, samplerSeq, matrixSeq, paramSeq) \
    GPT_CALLER_PROTO_2D(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o))

//================================================================
//
// GPUTOOL_2D_BEG_EX2
// GPUTOOL_2D_END_EX2
//
//================================================================

#define GPUTOOL_2D_BEG_EX2(prefix, tileSize, cellSize, superTaskCount, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPT_MAIN_2D_BEG(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o), PREP_ARG2_0 tileSize, PREP_ARG2_1 tileSize, \
        PREP_ARG2_0 cellSize, PREP_ARG2_1 cellSize, superTaskCount, keepAllThreads) \

#define GPUTOOL_2D_END_EX2 \
    GPT_MAIN_2D_END

//================================================================
//
// GPUTOOL_2D_BEG_EX
// GPUTOOL_2D_END_EX
//
//================================================================

#define GPUTOOL_2D_BEG_EX(prefix, tileSize, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPUTOOL_2D_BEG_EX2(prefix, tileSize, (1, 1), 1, keepAllThreads, samplerSeq, matrixSeq, paramSeq)

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
