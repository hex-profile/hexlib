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

KIT_CREATE1(GpuPropertiesKit, const GpuProperties&, gpuProperties);

//================================================================
//
// GpuFastAllocKit
//
//================================================================

KIT_CREATE1(GpuFastAllocKit, AllocatorObject<GpuAddrU>&, gpuFastAlloc);

//================================================================
//
// GpuCurrentContextKit
//
// Default GPU command queue.
//
//================================================================

struct GpuContext;

KIT_CREATE1(GpuCurrentContextKit, const GpuContext&, gpuCurrentContext);

//================================================================
//
// GpuCurrentStreamKit
//
// Default GPU command queue.
//
//================================================================

struct GpuStream;

KIT_CREATE1(GpuCurrentStreamKit, const GpuStream&, gpuCurrentStream);

//================================================================
//
// GpuTransferKit
//
//================================================================

struct GpuTransfer;

KIT_CREATE1(GpuTransferKit, GpuTransfer&, gpuTransfer);

//================================================================
//
// GpuStreamWaitingKit
//
//================================================================

struct GpuStreamWaiting;

KIT_CREATE1(GpuStreamWaitingKit, GpuStreamWaiting&, gpuStreamWaiting);

//================================================================
//
// GpuEventAllocKit
//
//================================================================

struct GpuEventAllocator;

KIT_CREATE1(GpuEventAllocKit, GpuEventAllocator&, gpuEventAlloc);

//================================================================
//
// GpuEventRecordingKit
//
//================================================================

struct GpuEventRecording;

KIT_CREATE1(GpuEventRecordingKit, GpuEventRecording&, gpuEventRecording);

//================================================================
//
// GpuEventWaitingKit
//
//================================================================

struct GpuEventWaiting;

KIT_CREATE1(GpuEventWaitingKit, GpuEventWaiting&, gpuEventWaiting);

//================================================================
//
// GpuKernelCallingKit
//
//================================================================

struct GpuKernelCalling;

KIT_CREATE1(GpuKernelCallingKit, GpuKernelCalling&, gpuKernelCalling);

//================================================================
//
// GpuSamplerSetupKit
//
//================================================================

struct GpuSamplerSetup;

KIT_CREATE1(GpuSamplerSetupKit, GpuSamplerSetup&, gpuSamplerSetting);

//================================================================
//
// GpuTextureAllocKit
//
//================================================================

struct GpuTextureAllocator;

KIT_CREATE1(GpuTextureAllocKit, GpuTextureAllocator&, gpuTextureAlloc);

//================================================================
//
// GpuAppExecKit
//
//================================================================

KIT_COMBINE9(GpuAppExecKitWithoutFastAlloc, GpuCurrentContextKit, GpuPropertiesKit, GpuCurrentStreamKit, GpuTransferKit, GpuSamplerSetupKit, GpuKernelCallingKit,
    GpuStreamWaitingKit, GpuEventRecordingKit, GpuEventWaitingKit);

KIT_COMBINE2(GpuAppExecKit, GpuAppExecKitWithoutFastAlloc, GpuFastAllocKit);

//================================================================
//
// GpuAppAllocKit
//
//================================================================

KIT_COMBINE2(GpuAppAllocKit, GpuTextureAllocKit, GpuEventAllocKit);

#define GPU_APP_ALLOC_KIT_LIST \
    (GpuTextureAllocKit) (GpuEventAllocKit)

//================================================================
//
// GpuAppFullKit
//
//================================================================

KIT_COMBINE2(GpuAppFullKit, GpuAppExecKit, GpuAppAllocKit);
