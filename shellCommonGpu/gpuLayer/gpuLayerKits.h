#pragma once

#include "kit/kit.h"
#include "gpuAppliedApi/gpuAppliedKits.h"

//================================================================
//
// GpuInitialization
//
//================================================================

struct GpuInitialization;
KIT_CREATE1(GpuInitializationKit, GpuInitialization&, gpuInitialization);

//================================================================
//
// GpuContextCreationKit
//
//================================================================

struct GpuContextCreation;
KIT_CREATE1(GpuContextCreationKit, GpuContextCreation&, gpuContextCreation);

//================================================================
//
// GpuModuleCreationKit
//
//================================================================

struct GpuModuleCreation;
KIT_CREATE1(GpuModuleCreationKit, GpuModuleCreation&, gpuModuleCreation);

//================================================================
//
// GpuKernelLoadingKit
//
//================================================================

struct GpuKernelLoading;
KIT_CREATE1(GpuKernelLoadingKit, GpuKernelLoading&, gpuKernelLoading);

//================================================================
//
// GpuSamplerLoadingKit
//
//================================================================

struct GpuSamplerLoading;
KIT_CREATE1(GpuSamplerLoadingKit, GpuSamplerLoading&, gpuSamplerLoading);

//================================================================
//
// GpuMemoryAllocationKit
//
//================================================================

struct GpuMemoryAllocation;
KIT_CREATE1(GpuMemoryAllocationKit, GpuMemoryAllocation&, gpuMemoryAllocation);

//================================================================
//
// GpuStreamCreationKit
//
//================================================================

struct GpuStreamCreation;
KIT_CREATE1(GpuStreamCreationKit, GpuStreamCreation&, gpuStreamCreation);

//================================================================
//
// GpuBenchmarkingControlKit
//
//================================================================

struct GpuBenchmarkingControl;
KIT_CREATE1(GpuBenchmarkingControlKit, GpuBenchmarkingControl&, gpuBenchmarkingControl);

//================================================================
//
// GpuInitKit
// GpuExecKit
//
//================================================================

KIT_COMBINE9(GpuInitKit, GpuInitializationKit, GpuContextCreationKit, GpuModuleCreationKit,
    GpuKernelLoadingKit, GpuSamplerLoadingKit, GpuMemoryAllocationKit, GpuTextureAllocKit, GpuStreamCreationKit, GpuEventAllocKit);

KIT_COMBINE7(GpuExecKit, GpuTransferKit, GpuSamplerSetupKit, GpuKernelCallingKit, GpuStreamWaitingKit,
    GpuEventRecordingKit, GpuEventWaitingKit, GpuBenchmarkingControlKit);
