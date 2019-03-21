#pragma once

#include "gpuDevice/gpuDevice.h"
#include "data/gpuPtr.h"

//================================================================
//
// EmptyKernelParams
//
//================================================================

struct EmptyKernelParams
{
};

//================================================================
//
// getEmptyKernelLink
//
//================================================================

#if !DEVCODE
const GpuKernelLink& getEmptyKernelLink();
#endif

//================================================================
//
// ReadMemoryKernelParams
//
// Works by 32-bit words
//
//================================================================

using ReadMemoryAtom = uint32;

////

struct ReadMemoryKernelParams
{
    GpuPtr(const ReadMemoryAtom) srcDataPtr;
    GpuPtr(ReadMemoryAtom) dstDataPtr;
    GpuAddrU atomCount;

    bool writeEnabled;
};

//================================================================
//
// getReadMemoryKernelLink
//
//================================================================

#if !DEVCODE
const GpuKernelLink& getReadMemoryKernelLink();
#endif
