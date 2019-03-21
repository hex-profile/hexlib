#pragma once

#include "gpuLayer/gpuLayerCuda/gpuLayerCuda.h"
#include "gpuLayer/gpuLayerEmu/gpuLayerEmu.h"

//================================================================
//
// GpuInitApiImpl
// GpuExecApiImpl
//
//================================================================

#if HEXLIB_PLATFORM == 0

    using GpuInitApiImpl = EmuInitApiThunk;
    using GpuExecApiImpl = EmuExecApiThunk;

#elif HEXLIB_PLATFORM == 1

    using GpuInitApiImpl = CudaInitApiThunk;
    using GpuExecApiImpl = CudaExecApiThunk;

#else

    #error

#endif
