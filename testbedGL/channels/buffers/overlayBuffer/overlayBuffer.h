#pragma once

#include "storage/smartPtr.h"

#include "imageConsole/gpuBaseConsole.h"
#include "gpuLayer/gpuLayerKits.h"
#include "storage/adapters/callable.h"

namespace overlayBuffer {

//================================================================
//
// Type
//
//================================================================

using Type = uint8_x4;

//================================================================
//
// ImageUser
//
//================================================================

using ImageUser = Callable<stdbool (bool valid, const GpuMatrix<const Type>& image, stdNullPars)>;

//================================================================
//
// OverlayBuffer
//
//================================================================

struct OverlayBuffer
{

    static UniquePtr<OverlayBuffer> create();
    virtual ~OverlayBuffer() {}

    virtual void clearMemory() =0;

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void clearImage() =0;

    using SetImageKit = KitCombine<GpuProcessKit, GpuEventAllocKit, GpuMemoryAllocationKit>; // to allocate events
    virtual stdbool setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(SetImageKit)) =0;

    ////

    using UseImageKit = GpuProcessKit;
    virtual stdbool useImage(ImageUser& imageUser, stdPars(UseImageKit)) =0;

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const =0;
    virtual void reset() =0;
    virtual bool absorb(OverlayBuffer& other) =0;
    virtual void moveFrom(OverlayBuffer& other) =0;

};

//----------------------------------------------------------------

}

using overlayBuffer::OverlayBuffer;
