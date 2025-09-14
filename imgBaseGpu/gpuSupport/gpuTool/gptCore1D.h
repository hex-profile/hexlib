#pragma once

#include "gptCommonImpl.h"

//================================================================
//
// GPT 1D caller tools
//
//================================================================

#define GPT_EXPOSE_ARRAY(Type, name) \
    auto name = (o.name##Ptr + (index)); \
    MAKE_VARIABLE_USED(name);

#define GPT_DECLARE_ARRAY_ARG(Type, name) \
    const GpuArray<Type>& name##Array,

////

#define GPT_GET_ARRAY_SIZE(Type, name) \
    globSize = name##Array.size();

#define GPT_CHECK_ARRAY_SIZE(Type, name) \
    REQUIRE(globSize == name##Array.size());

////

#define GPT_SET_ARRAY_FIELD(Type, name) \
    ARRAY_EXPOSE_EX(name##Array, name); \
    kernelParams.name##Ptr = name##Ptr; \

//================================================================
//
// GPT_DECLARE_ARRAY_PARAMS
//
//================================================================

#define GPT_DECLARE_ARRAY(Type, name) \
    GpuArrayPtr(Type) name##Ptr; \

#define GPT_DECLARE_ARRAY_PARAMS(prefix, GlobSizeType, samplerList, arrayList, paramList) \
    \
    struct prefix##Params \
    { \
        GlobSizeType globSize; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_PARAM, o) \
        GPT_FOREACH(arrayList, GPT_DECLARE_ARRAY) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM) \
    };

//================================================================
//
// GPT_MAKE_KERNEL_1D_*
//
//================================================================

#define GPT_MAKE_KERNEL_1D_BEG(prefix, samplerList, arrayList, paramList, tileSizep, superTaskSupport, keepAllThreads) \
    \
    devDefineKernel(prefix##Kernel, prefix##Params, o) \
    { \
        constexpr Space vTileSize = (tileSizep); \
        COMPILE_ASSERT(vTileSize >= 1); \
        \
        const Space vTileMember = devThreadX; \
        \
        const Space vTileIdx = devGroupX; \
        const Space vTileOrg = vTileIdx * vTileSize; \
        \
        const Space index = (vTileSize == 1) ? devGroupX : (vTileOrg + vTileMember); \
        \
        PREP_IF(superTaskSupport, const Space vSuperTaskCount = devGroupCountZ; MAKE_VARIABLE_USED(vSuperTaskCount);) \
        PREP_IF(superTaskSupport, const Space vSuperTaskIdx = devGroupZ; MAKE_VARIABLE_USED(vSuperTaskIdx);) \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_EXPOSE_SAMPLER, prefix) \
        GPT_FOREACH(arrayList, GPT_EXPOSE_ARRAY) \
        GPT_FOREACH(paramList, GPT_EXPOSE_PARAM) \
        \
        const auto& vGlobSize = o.globSize; \
        \
        const bool vItemIsActive = \
            ((vTileSize == 1) || (SpaceU(index) < SpaceU(vGlobSize))); \
        \
        if_not (keepAllThreads) \
            if_not (vItemIsActive) \
                return; \
        \
        {

#define GPT_MAKE_KERNEL_1D_END \
        } \
    }

//================================================================
//
// GPT_MAKE_CALLER_1D
//
//================================================================

#define GPT_MAKE_CALLER_1D(prefix, samplerList, arrayList, paramList, tileSize, superTaskSupport) \
    \
    hostDeclareKernel(prefix##Kernel, prefix##Params, o) \
    \
    stdbool prefix \
    ( \
        PREP_IF(superTaskSupport, Space superTaskCount) PREP_IF_COMMA(superTaskSupport) \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(arrayList, GPT_DECLARE_ARRAY_ARG) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \
    { \
        if_not (kit.dataProcessing) \
            returnTrue; \
        \
        auto globSize = Space{0}; \
        GPT_FOREACH(arrayList, GPT_GET_ARRAY_SIZE) \
        \
        GPT_FOREACH(arrayList, GPT_CHECK_ARRAY_SIZE) \
        \
        PREP_IF(PREP_NOT(superTaskSupport), constexpr Space superTaskCount = 1;) \
        \
        Space groupCount = divUpNonneg(globSize, tileSize); \
        REQUIRE(superTaskCount >= 1); \
        \
        if_not (groupCount && superTaskCount) \
            returnTrue; \
        \
        GPT_FOREACH_SAMPLER(samplerList, GPT_BIND_SAMPLER, prefix) \
        \
        prefix##Params kernelParams; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_SET_SAMPLER_FIELD, o) \
        GPT_FOREACH(arrayList, GPT_SET_ARRAY_FIELD) \
        kernelParams.globSize = globSize; \
        GPT_FOREACH(paramList, GPT_SET_PARAM_FIELD) \
        \
        require \
        ( \
            kit.gpuKernelCalling.callKernel \
            ( \
                point3D(groupCount, 1, superTaskCount), \
                point(tileSize, 1), \
                globSize, \
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
// GPT_CALLER_1D_PROTO
//
//================================================================

#define GPT_CALLER_1D_PROTO(prefix, samplerList, arrayList, paramList) \
    \
    stdbool prefix \
    ( \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_ARG, o) \
        GPT_FOREACH(arrayList, GPT_DECLARE_ARRAY_ARG) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM_ARG) \
        stdPars(GpuProcessKit) \
    ) \

//================================================================
//
// GPT_MAIN_1D_BEG
// GPT_MAIN_1D_END
//
//================================================================

#define GPT_MAIN_1D_BEG(prefix, samplerList, arrayList, paramList, tileSize, superTaskSupport, keepAllThreads) \
    GPT_DECLARE_ARRAY_PARAMS(prefix, Space, samplerList, arrayList, paramList) \
    DEV_ONLY(GPT_DEFINE_SAMPLERS(prefix, samplerList)) \
    HOST_ONLY(GPT_MAKE_CALLER_1D(prefix, samplerList, arrayList, paramList, tileSize, superTaskSupport)) \
    DEV_ONLY(GPT_MAKE_KERNEL_1D_BEG(prefix, samplerList, arrayList, paramList, tileSize, superTaskSupport, keepAllThreads))

#define GPT_MAIN_1D_END \
    DEV_ONLY(GPT_MAKE_KERNEL_1D_END)

//================================================================
//
// GPUTOOL_1D_PROTO
//
//================================================================

#define GPUTOOL_1D_PROTO(prefix, samplerSeq, matrixSeq, paramSeq) \
    GPT_CALLER_1D_PROTO(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o))

//================================================================
//
// GPUTOOL_1D_BEG_EX2
// GPUTOOL_1D_END
//
//================================================================

#define GPUTOOL_1D_BEG_EX2(prefix, tileSize, superTaskSupport, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPT_MAIN_1D_BEG(prefix, samplerSeq (o), matrixSeq (o), paramSeq (o), tileSize, \
        superTaskSupport, keepAllThreads) \

#define GPUTOOL_1D_END \
    GPT_MAIN_1D_END

//================================================================
//
// GPUTOOL_1D_BEG_EX
//
//================================================================

#define GPUTOOL_1D_BEG_EX(prefix, tileSize, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    GPUTOOL_1D_BEG_EX2(prefix, tileSize, PREP_FALSE, keepAllThreads, samplerSeq, matrixSeq, paramSeq)

#define GPUTOOL_1D_EX(prefix, tileSize, keepAllThreads, samplerSeq, matrixSeq, paramSeq, iterationBody) \
    GPUTOOL_1D_BEG_EX(prefix, tileSize, keepAllThreads, samplerSeq, matrixSeq, paramSeq) \
    DEV_ONLY(iterationBody) \
    GPUTOOL_1D_END

//================================================================
//
// Default 1D GPU block size.
//
//================================================================

#define GPUTOOL_1D_DEFAULT_THREAD_COUNT 256

//================================================================
//
// GPUTOOL_1D
// GPUTOOL_1D_BEG
// GPUTOOL_1D_END
//
//================================================================

#define GPUTOOL_1D_BEG(prefix, samplerSeq, matrixSeq, paramSeq) \
    GPUTOOL_1D_BEG_EX(prefix, GPUTOOL_1D_DEFAULT_THREAD_COUNT, false, samplerSeq, matrixSeq, paramSeq)

#define GPUTOOL_1D(prefix, samplerSeq, matrixSeq, paramSeq, iterationBody) \
    GPUTOOL_1D_BEG_EX(prefix, GPUTOOL_1D_DEFAULT_THREAD_COUNT, false, samplerSeq, matrixSeq, paramSeq) \
    DEV_ONLY(iterationBody) \
    GPUTOOL_1D_END
