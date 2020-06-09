#pragma once

#include "gptCommonImpl.h"

//================================================================
//
// GPT_PLAIN_MAKE_KERNEL_BEG
// GPT_PLAIN_MAKE_KERNEL_END
//
//================================================================

#define GPT_PLAIN_MAKE_KERNEL_BEG(prefix, samplerList, paramList, chunkSizep, keepAllThreads) \
    \
    devDefineKernel(prefix##Kernel, prefix##Params, o) \
    { \
        GPT_FOREACH_SAMPLER(samplerList, GPT_EXPOSE_SAMPLER, prefix) \
        GPT_FOREACH(paramList, GPT_EXPOSE_PARAM) \
        \
        const Space GROUP_FLAT_SIZE = (chunkSizep); \
        COMPILE_ASSERT(GROUP_FLAT_SIZE >= 1); \
        \
        const Space groupFlatMember = devThreadX; \
        MAKE_VARIABLE_USED(groupFlatMember); \
        \
        const Space CHUNK_SIZE_ = (chunkSizep); \
        const Space chunkMember_ = devThreadX; \
        \
        Space plainGlobalIdx = (CHUNK_SIZE_ == 1) ? devGroupX : (devGroupX * CHUNK_SIZE_ + chunkMember_); \
        \
        Space plainGlobalSize = o.globSize; \
        MAKE_VARIABLE_USED(plainGlobalSize); \
        \
        bool itemIsActive_ = \
            ((CHUNK_SIZE_ == 1) || (SpaceU(plainGlobalIdx) < SpaceU(plainGlobalSize))); \
        \
        if_not (keepAllThreads) \
            if_not (itemIsActive_) \
                return; \
        \
        {

#define GPT_PLAIN_MAKE_KERNEL_END \
        } \
    }

//================================================================
//
// GPT_PLAIN_MAKE_CALLER
//
//================================================================

#define GPT_PLAIN_MAKE_CALLER(prefix, samplerList, paramList, chunkSize) \
    \
    hostDeclareKernel(prefix##Kernel, prefix##Params, o); \
    \
    stdbool prefix \
    ( \
        Space globSize, \
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
        stdEnterElemCount(globSize); \
        \
        REQUIRE(globSize >= 0); \
        \
        Space groupCount = divUpNonneg(globSize, chunkSize); \
        \
        if_not (groupCount != 0) \
            returnTrue; \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_BIND_SAMPLER, prefix) \
        \
        prefix##Params kernelParams; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_SET_SAMPLER_FIELD, o) \
        kernelParams.globSize = globSize; \
        GPT_FOREACH(paramList, GPT_SET_PARAM_FIELD) \
        \
        require \
        ( \
            kit.gpuKernelCalling.callKernel \
            ( \
                point(groupCount, 1), \
                point((chunkSize), 1), \
                globSize, \
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
// GPT_PLAIN_CALLER_PROTO
//
//================================================================

#define GPT_PLAIN_CALLER_PROTO(prefix, samplerList, paramList) \
    \
    stdbool prefix \
    ( \
        Space globSize, \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \

//================================================================
//
// GPT_PLAIN_MAIN_BEG
// GPT_PLAIN_MAIN_END
//
//================================================================

#define GPT_PLAIN_MAIN_BEG(prefix, samplerList, paramList, chunkSize, keepAllThreads) \
    GPT_DECLARE_PARAMS(prefix, Space, samplerList, (o), paramList) \
    DEV_ONLY(GPT_DEFINE_SAMPLERS(prefix, samplerList)) \
    DEV_ONLY(GPT_PLAIN_MAKE_KERNEL_BEG(prefix, samplerList, paramList, chunkSize, keepAllThreads)) \
    HOST_ONLY(GPT_PLAIN_MAKE_CALLER(prefix, samplerList, paramList, chunkSize))

#define GPT_PLAIN_MAIN_END \
    DEV_ONLY(GPT_PLAIN_MAKE_KERNEL_END)

//================================================================
//
// GPUTOOL_PLAIN_BEG
// GPUTOOL_PLAIN_END
// GPUTOOL_PLAIN_PROTO
//
//================================================================

#define GPUTOOL_PLAIN_BEG(prefix, chunkSize, keepAllThreads, samplerSeq, paramSeq) \
    GPT_PLAIN_MAIN_BEG(prefix, samplerSeq (o), paramSeq (o), chunkSize, keepAllThreads)

#define GPUTOOL_PLAIN_END \
    GPT_PLAIN_MAIN_END

#define GPUTOOL_PLAIN_PROTO(prefix, samplerSeq, paramSeq) \
    GPT_PLAIN_CALLER_PROTO(prefix, samplerSeq (o), paramSeq (o))
