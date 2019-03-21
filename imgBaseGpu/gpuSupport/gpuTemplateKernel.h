#pragma once

#include "gpuSupport/gpuMixedCode.h"
#include "gpuDevice/gpuDevice.h"
#include "prepTools/prepList.h"

//================================================================
//
// GTK_TEMPLATE_DECL
//
//================================================================

#define GTK_TEMPLATE_DECL(templateList) \
    PREP_LIST_ENUM_PAIR(templateList, GTK_TEMPLATE_DECL0, o)

#define GTK_TEMPLATE_DECL0(Type, name, o) \
    Type name

//----------------------------------------------------------------

#define GTK_TEMPLATE_PAR(specList) \
    PREP_LIST_ENUM(specList, GTK_TEMPLATE_PAR0, o)

#define GTK_TEMPLATE_PAR0(value, o) \
    value

//----------------------------------------------------------------

#define GTK_TEMPLATE_MANGLE(specList) \
    PREP_LIST_ENUM(specList, GTK_TEMPLATE_MANGLE0, o)

#define GTK_TEMPLATE_MANGLE0(value, o) \
    value

//================================================================
//
// GTK_HOST
//
//================================================================

#define GTK_HOST(templateList, kernelName, KernelParams, kernelLink, specName, specList) \
    \
    template <GTK_TEMPLATE_DECL(templateList)> \
    inline const GpuKernelLink& kernelLink(); \
    \
    template <> \
    inline const GpuKernelLink& kernelLink<GTK_TEMPLATE_PAR(specList)>() \
        {return specName;} \

//================================================================
//
// GTK_DEVICE
//
//================================================================

#define GTK_DEVICE(templateList, kernelName, KernelParams, specName, specList) \
    \
    devDefineKernel(specName, KernelParams<GTK_TEMPLATE_PAR(specList)>, o) \
        {kernelName(o, devPass);}

//================================================================
//
// GTK_INSTANTIATE
//
//================================================================

#define GTK_INSTANTIATE(templateList, kernelName, KernelParams, kernelLink, specName, specList) \
    DEV_ONLY(GTK_DEVICE(templateList, kernelName, KernelParams, specName, specList)) \
    HOST_ONLY(GTK_HOST(templateList, kernelName, KernelParams, kernelLink, specName, specList))

//================================================================
//
// GPU_TEMPLATE_KERNEL_INST
//
//================================================================

#define GPU_TEMPLATE_KERNEL_INST(templateList, kernelName, KernelParams, kernelLink, specName, specList) \
    GTK_INSTANTIATE(templateList (o), kernelName, KernelParams, kernelLink, specName, specList (o))
