#pragma once

#include "kit/kit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "data/gpuAddr.h"

//================================================================
//
// GpuProperties
//
//================================================================

struct GpuProperties;

KIT_CREATE(GpuPropertiesKit, const GpuProperties&, gpuProperties);

//================================================================
//
// GpuFastAllocKit
//
//================================================================

KIT_CREATE(GpuFastAllocKit, AllocatorInterface<GpuAddrU>&, gpuFastAlloc);

//================================================================
//
// GpuCurrentContextKit
//
// Default GPU command queue.
//
//================================================================

struct GpuContext;

KIT_CREATE(GpuCurrentContextKit, const GpuContext&, gpuCurrentContext);

//================================================================
//
// GpuCurrentStreamKit
//
// Default GPU command queue.
//
//================================================================

struct GpuStream;

KIT_CREATE(GpuCurrentStreamKit, const GpuStream&, gpuCurrentStream);

//================================================================
//
// GpuTransferKit
//
//================================================================

struct GpuTransfer;

KIT_CREATE(GpuTransferKit, GpuTransfer&, gpuTransfer);

//================================================================
//
// GpuStreamWaitingKit
//
//================================================================

struct GpuStreamWaiting;

KIT_CREATE(GpuStreamWaitingKit, GpuStreamWaiting&, gpuStreamWaiting);

//================================================================
//
// GpuEventAllocKit
//
//================================================================

struct GpuEventAllocator;

KIT_CREATE(GpuEventAllocKit, GpuEventAllocator&, gpuEventAlloc);

//================================================================
//
// GpuEventRecordingKit
//
//================================================================

struct GpuEventRecording;

KIT_CREATE(GpuEventRecordingKit, GpuEventRecording&, gpuEventRecording);

//================================================================
//
// GpuEventWaitingKit
//
//================================================================

struct GpuEventWaiting;

KIT_CREATE(GpuEventWaitingKit, GpuEventWaiting&, gpuEventWaiting);

//================================================================
//
// GpuKernelCallingKit
//
//================================================================

struct GpuKernelCalling;

KIT_CREATE(GpuKernelCallingKit, GpuKernelCalling&, gpuKernelCalling);

//================================================================
//
// GpuSamplerSetupKit
//
//================================================================

struct GpuSamplerSetup;

KIT_CREATE(GpuSamplerSetupKit, GpuSamplerSetup&, gpuSamplerSetting);

//================================================================
//
// GpuTextureAllocKit
//
//================================================================

struct GpuTextureAllocator;

KIT_CREATE(GpuTextureAllocKit, GpuTextureAllocator&, gpuTextureAlloc);

//================================================================
//
// GpuAppExecKit
//
//================================================================

using GpuAppExecKitWithoutFastAlloc = KitCombine<GpuCurrentContextKit, GpuPropertiesKit, GpuCurrentStreamKit, GpuTransferKit, GpuSamplerSetupKit, GpuKernelCallingKit,
    GpuStreamWaitingKit, GpuEventRecordingKit, GpuEventWaitingKit>;

using GpuAppExecKit = KitCombine<GpuAppExecKitWithoutFastAlloc, GpuFastAllocKit>;

//================================================================
//
// GpuAppAllocKit
//
//================================================================

using GpuAppAllocKit = KitCombine<GpuTextureAllocKit, GpuEventAllocKit>;

#define GPU_APP_ALLOC_KIT_LIST \
    (GpuTextureAllocKit) (GpuEventAllocKit)

//================================================================
//
// GpuAppFullKit
//
//================================================================

using GpuAppFullKit = KitCombine<GpuAppExecKit, GpuAppAllocKit>;
