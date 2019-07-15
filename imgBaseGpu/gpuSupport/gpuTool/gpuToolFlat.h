#pragma once

#include "gptCommonImpl.h"

//================================================================
//
// GPT_FLAT_MAKE_KERNEL_BEG
// GPT_FLAT_MAKE_KERNEL_END
//
//================================================================

#define GPT_FLAT_MAKE_KERNEL_BEG(prefix, samplerList, paramList, groupSizeXp, groupSizeYp) \
    \
    devDefineKernel(prefix##Kernel, prefix##Params, o) \
    { \
        GPT_FOREACH_SAMPLER(samplerList, GPT_EXPOSE_SAMPLER, prefix) \
        GPT_FOREACH(paramList, GPT_EXPOSE_PARAM) \
        \
        const Space GROUP_SIZE_X = (groupSizeXp); \
        const Space GROUP_SIZE_Y = (groupSizeYp); \
        Point<Space> groupSize = point(GROUP_SIZE_X, GROUP_SIZE_Y); \
        MAKE_VARIABLE_USED(groupSize); \
        \
        const Space GROUP_FLAT_SIZE = GROUP_SIZE_X * GROUP_SIZE_Y; \
        COMPILE_ASSERT(GROUP_FLAT_SIZE >= 1); \
        \
        const Space groupFlatMember = devThreadX + devThreadY * GROUP_SIZE_X; \
        MAKE_VARIABLE_USED(groupFlatMember); \
        \
        const Space groupFlatIdx = devGroupX + devGroupY * devGroupCountX; \
        MAKE_VARIABLE_USED(groupFlatIdx); \
        \
        {

#define GPT_FLAT_MAKE_KERNEL_END \
        } \
    }

//================================================================
//
// GPT_FLAT_MAKE_CALLER
//
//================================================================

#define GPT_FLAT_MAKE_CALLER(prefix, samplerList, paramList, groupSizeX, groupSizeY) \
    \
    hostDeclareKernel(prefix##Kernel, prefix##Params, o); \
    \
    stdbool prefix \
    ( \
        const Point<Space>& groupCount, \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \
    { \
        stdScopedBegin; \
        \
        if_not (kit.dataProcessing) \
            returnTrue; \
        \
        uint32 elemCount = areaOf(groupCount) * (groupSizeX) * (groupSizeY); \
        stdEnterElemCount(elemCount); \
        \
        if_not (groupCount.X > 0 && groupCount.Y > 0) \
            returnTrue; \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_BIND_SAMPLER, prefix) \
        \
        prefix##Params kernelParams; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_SET_SAMPLER_FIELD, o) \
        GPT_FOREACH(paramList, GPT_SET_PARAM_FIELD) \
        \
        require \
        ( \
            kit.gpuKernelCalling.callKernel \
            ( \
                groupCount, \
                point((groupSizeX), (groupSizeY)), \
                elemCount, \
                prefix##Kernel, \
                kernelParams, \
                kit.gpuCurrentStream, \
                stdPassLocationMsg("Kernel") \
            ) \
        ); \
        \
        stdScopedEnd; \
    } \

//================================================================
//
// GPT_FLAT_CALLER_PROTO
//
//================================================================

#define GPT_FLAT_CALLER_PROTO(prefix, samplerList, paramList) \
    \
    stdbool prefix \
    ( \
        const Point<Space>& groupCount, \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \

//================================================================
//
// GPT_FLAT_MAIN_BEG
// GPT_FLAT_MAIN_END
//
//================================================================

#define GPT_FLAT_MAIN_BEG(prefix, samplerList, paramList, groupSizeX, groupSizeY) \
    GPT_DECLARE_PARAMS(prefix, Space, samplerList, (o), paramList) \
    GPT_DEFINE_SAMPLERS(prefix, samplerList) \
    HOST_ONLY(GPT_FLAT_MAKE_CALLER(prefix, samplerList, paramList, groupSizeX, groupSizeY)) \
    DEV_ONLY(GPT_FLAT_MAKE_KERNEL_BEG(prefix, samplerList, paramList, groupSizeX, groupSizeY))

#define GPT_FLAT_MAIN_END \
    DEV_ONLY(GPT_FLAT_MAKE_KERNEL_END)

//================================================================
//
// GPUTOOL_FLAT_BEG
// GPUTOOL_FLAT_END
// GPUTOOL_FLAT_PROTO
//
//================================================================

#define GPUTOOL_FLAT_BEG(prefix, groupSize, samplerSeq, paramSeq) \
    GPT_FLAT_MAIN_BEG(prefix, samplerSeq (o), paramSeq (o), PREP_ARG2_0 groupSize, PREP_ARG2_1 groupSize)

#define GPUTOOL_FLAT_END \
    GPT_FLAT_MAIN_END

#define GPUTOOL_FLAT_PROTO(prefix, samplerSeq, paramSeq) \
    GPT_FLAT_CALLER_PROTO(prefix, samplerSeq (o), paramSeq (o))
