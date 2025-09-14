#pragma once

#include "kit/kit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "data/gpuAddr.h"

//================================================================
//
// OpaqueStruct
//
//================================================================

template <size_t size, unsigned hash>
class OpaqueStruct;

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
// Current GPU context.
//
//================================================================

using GpuContext = OpaqueStruct<8, 0xAD23E3A0u>;

KIT_CREATE(GpuCurrentContextKit, const GpuContext&, gpuCurrentContext);

//================================================================
//
// GpuCurrentStreamKit
//
// Current GPU command queue.
//
//================================================================

using GpuStream = OpaqueStruct<8, 0x98F6A9F0u>;

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
// GpuCountingPhaseProhibitionControlKit
//
//================================================================

struct GpuCountingPhaseProhibitionControl;

KIT_CREATE(GpuCountingPhaseProhibitionControlKit, GpuCountingPhaseProhibitionControl&, gpuCountingPhaseProhibitionControl);

//================================================================
//
// GpuAppExecKit
//
//================================================================

using GpuAppExecKitWithoutFastAlloc = KitCombine
<
    GpuCurrentContextKit,
    GpuPropertiesKit,
    GpuCurrentStreamKit,
    GpuTransferKit,
    GpuSamplerSetupKit,
    GpuKernelCallingKit,
    GpuStreamWaitingKit,
    GpuEventRecordingKit,
    GpuEventWaitingKit,
    GpuCountingPhaseProhibitionControlKit
>;

using GpuAppExecKit = KitCombine<GpuAppExecKitWithoutFastAlloc, GpuFastAllocKit>;

//================================================================
//
// GpuAppAllocKit
//
//================================================================

using GpuAppAllocKit = KitCombine<GpuTextureAllocKit, GpuEventAllocKit>;

//================================================================
//
// GpuAppFullKit
//
//================================================================

using GpuAppFullKit = KitCombine<GpuAppExecKit, GpuAppAllocKit>;
