#pragma once

#include "gpuSupport/gpuMixedCode.h"
#include "gpuDevice/devSampler/devSampler.h"
#include "imageRead/interpType.h"
#include "imageRead/borderMode.h"
#include "prepTools/prepList.h"
#include "vectorTypes/vectorType.h"
#include "gpuDevice/gpuDevice.h"
#include "data/gpuMatrix.h"

#if HOSTCODE
#include "numbers/divRound.h"
#include "gpuProcessKit.h"
#include "errorLog/errorLog.h"
#include "numbers/float/floatType.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#endif

//================================================================
//
// GPT_FOREACH
//
//================================================================

#define GPT_FOREACH(list, action) \
    PREP_LIST_FOREACH_PAIR(list, GPT_FOREACH_ACTION, action)

#define GPT_FOREACH_ACTION(Type, name, action) \
    action(Type, name)

//================================================================
//
// GPT_ENUM
//
//================================================================

#define GPT_ENUM(list, action) \
    PREP_LIST_ENUM_PAIR(list, action)

//================================================================
//
// GPT_FOREACH_SAMPLER
//
//================================================================

#define GPT_FOREACH_SAMPLER(list, action, extra) \
    PREP_LIST_FOREACH(list, GPT_FOREACH_SAMPLER0, (action, extra))

#define GPT_FOREACH_SAMPLER0(item, param) \
    GPT_FOREACH_SAMPLER1(item, PREP_ARG2_0 param, PREP_ARG2_1 param)

#define GPT_FOREACH_SAMPLER1(item, action, extra) \
    GPT_FOREACH_SAMPLER2(action, PREP_ARG4_0 item, PREP_ARG4_1 item, PREP_ARG4_2 item, PREP_ARG4_3 item, extra)

#define GPT_FOREACH_SAMPLER2(action, Type, name, interp, border, extra) \
    action(Type, name, interp, border, extra)

//================================================================
//
// GPT_DECLARE_PARAMS
//
//================================================================

#define GPT_DECLARE_SAMPLER_PARAM(Type, name, interp, border, o) \
    Point<float32> name##Texstep;

#define GPT_DECLARE_MATRIX(Type, name) \
    GpuMatrixPtr(Type) name##MemPtr; \
    Space name##MemPitch;

#define GPT_DECLARE_PARAM(Type, name) \
    Type name;

#define GPT_DECLARE_PARAMS(prefix, GlobSizeType, samplerList, matrixList, paramList) \
    \
    struct prefix##Params \
    { \
        GlobSizeType globSize; \
        GPT_FOREACH_SAMPLER(samplerList, GPT_DECLARE_SAMPLER_PARAM, o) \
        GPT_FOREACH(matrixList, GPT_DECLARE_MATRIX) \
        GPT_FOREACH(paramList, GPT_DECLARE_PARAM) \
    };

//================================================================
//
// GPT_DEFINE_SAMPLERS
//
//================================================================

#define GPT_DEFINE_SAMPLERS(prefix, samplerList) \
    GPT_FOREACH_SAMPLER(samplerList, GPT_DEFINE_SAMPLER, prefix)

#define GPT_DEFINE_SAMPLER(Type, name, interp, border, prefix) \
    devDefineSampler(PREP_PASTE3(prefix, name, Sampler), DevSampler2D, DevSamplerFloat, VectorTypeRank<Type>::val)

//================================================================
//
// GPT_MAKE_SAMPLER_USED
//
//================================================================

#if defined(__CUDA_ARCH__)

    #define GPT_MAKE_SAMPLER_USED(sampler) \
        MAKE_VARIABLE_USED_BY_VALUE(sampler)

#else

    #define GPT_MAKE_SAMPLER_USED(sampler)

#endif

//================================================================
//
// GPT_EXPOSE_*
//
//================================================================

#define GPT_EXPOSE_SAMPLER(Type, name, interp, border, prefix) \
    const Point<float32>& name##Texstep = o.name##Texstep; MAKE_VARIABLE_USED(name##Texstep); \
    const InterpType name##Interp = (interp); MAKE_VARIABLE_USED(name##Interp); \
    const BorderMode name##Border = (border); MAKE_VARIABLE_USED(name##Border); \
    auto name##Sampler = PREP_PASTE3(prefix, name, Sampler); GPT_MAKE_SAMPLER_USED(name##Sampler);

#define GPT_EXPOSE_MATRIX(Type, name) \
    GpuMatrixPtr(Type) name = MATRIX_POINTER(o.name, X, Y); MAKE_VARIABLE_USED(name);

#define GPT_EXPOSE_PARAM(Type, name) \
    ParamType<Type>::T name = o.name; MAKE_VARIABLE_USED(name);

//================================================================
//
// GPT caller tools
//
//================================================================

#define GPT_DECLARE_SAMPLER_ARG(Type, name, interp, border, o) \
    const GpuMatrix<Type>& name##Matrix,

#define GPT_DECLARE_MATRIX_ARG(Type, name) \
    const GpuMatrix<Type>& name##Matrix,

#define GPT_DECLARE_PARAM_ARG(Type, name) \
    ParamType<Type>::T name,

//----------------------------------------------------------------

#define GPT_GET_SIZE(Type, name) \
    globSize = name##Matrix.size();

#define GPT_CHECK_SIZE(Type, name) \
    REQUIRE(globSize == name##Matrix.size());

#define GPT_BIND_SAMPLER(Type, name, interp, border, prefix) \
    require(kit.gpuSamplerSetting.setSamplerImage<Type>(PREP_PASTE3(prefix, name, Sampler), name##Matrix, border, interp == INTERP_LINEAR, true, true, \
        stdPassLocationMsg(PREP_STRINGIZE(PREP_PASTE2(name, Sampler)))));

#define GPT_SET_SAMPLER_FIELD(Type, name, interp, border, o) \
    kernelParams.name##Texstep = 1.f / convertFloat32(clampMin(name##Matrix.size(), 1));

#define GPT_SET_MATRIX_FIELD(Type, name) \
    MATRIX_EXPOSE_EX(name##Matrix, name); \
    kernelParams.name##MemPtr = name##MemPtr; \
    kernelParams.name##MemPitch = name##MemPitch;

#define GPT_SET_PARAM_FIELD(Type, name) \
    kernelParams.name = name;
