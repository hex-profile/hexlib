#pragma once

#include "kit/kit.h"
#include "gpuAppliedApi/gpuAppliedKits.h"

//================================================================
//
// GpuInitialization
//
//================================================================

struct GpuInitialization;
KIT_CREATE(GpuInitializationKit, GpuInitialization&, gpuInitialization);

//================================================================
//
// GpuContextCreationKit
//
//================================================================

struct GpuContextCreation;
KIT_CREATE(GpuContextCreationKit, GpuContextCreation&, gpuContextCreation);

//================================================================
//
// GpuContextSettingKit
//
//================================================================

struct GpuContextSetting;
KIT_CREATE(GpuContextSettingKit, GpuContextSetting&, gpuContextSetting);

//================================================================
//
// GpuModuleCreationKit
//
//================================================================

struct GpuModuleCreation;
KIT_CREATE(GpuModuleCreationKit, GpuModuleCreation&, gpuModuleCreation);

//================================================================
//
// GpuKernelLoadingKit
//
//================================================================

struct GpuKernelLoading;
KIT_CREATE(GpuKernelLoadingKit, GpuKernelLoading&, gpuKernelLoading);

//================================================================
//
// GpuSamplerLoadingKit
//
//================================================================

struct GpuSamplerLoading;
KIT_CREATE(GpuSamplerLoadingKit, GpuSamplerLoading&, gpuSamplerLoading);

//================================================================
//
// GpuMemoryAllocationKit
//
//================================================================

struct GpuMemoryAllocation;
KIT_CREATE(GpuMemoryAllocationKit, GpuMemoryAllocation&, gpuMemoryAllocation);

//================================================================
//
// GpuStreamCreationKit
//
//================================================================

struct GpuStreamCreation;
KIT_CREATE(GpuStreamCreationKit, GpuStreamCreation&, gpuStreamCreation);

//================================================================
//
// GpuBenchmarkingControlKit
//
//================================================================

struct GpuBenchmarkingControl;
KIT_CREATE(GpuBenchmarkingControlKit, GpuBenchmarkingControl&, gpuBenchmarkingControl);

//================================================================
//
// GpuInitKit
// GpuExecKit
//
//================================================================

using GpuInitKit = KitCombine
<
    GpuInitializationKit,
    GpuContextCreationKit,
    GpuContextSettingKit,
    GpuModuleCreationKit,
    GpuKernelLoadingKit,
    GpuSamplerLoadingKit,
    GpuMemoryAllocationKit,
    GpuTextureAllocKit,
    GpuStreamCreationKit,
    GpuEventAllocKit
>;

using GpuExecKit = KitCombine
<
    GpuTransferKit,
    GpuSamplerSetupKit,
    GpuKernelCallingKit,
    GpuStreamWaitingKit,
    GpuEventRecordingKit,
    GpuEventWaitingKit,
    GpuBenchmarkingControlKit,
    GpuCountingPhaseProhibitionControlKit
>;
