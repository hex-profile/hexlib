#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"
#include "vectorTypes/vectorBase.h"
#include "channels/buffers/logBuffer/logBuffer.h"
#include "channels/guiService/guiService.h"
#include "lib/eventReceivers.h"

namespace gui {

//================================================================
//
// RedrawRequest
//
//================================================================

struct RedrawRequest
{
    bool on = false;
};

//================================================================
//
// GuiModule
//
//================================================================

struct GuiModule
{
    static UniquePtr<GuiModule> create();
    virtual ~GuiModule() {}

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit) =0;

    virtual void extendMaxImageSize(const Point<Space>& size) =0;

    virtual OptionalObject<Point<Space>> getOverlayOffset() const =0;

    //----------------------------------------------------------------
    //
    // Realloc.
    //
    //----------------------------------------------------------------

    using ReallocKit = KitCombine<GpuModuleReallocKit, GpuAppAllocKit>;

    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(ReallocKit)) =0;

    //----------------------------------------------------------------
    //
    // Check wake.
    //
    //----------------------------------------------------------------

    struct CheckWakeArgs
    {
        LogBufferReading& globalLog;
    };

    using CheckWakeKit = KitCombine<ErrorLogKit, TimerKit>;

    virtual stdbool checkWake(const CheckWakeArgs& args, stdPars(CheckWakeKit)) =0;

    ////

    virtual OptionalObject<TimeMoment> getWakeMoment() const =0;

    //----------------------------------------------------------------
    //
    // Events.
    //
    //----------------------------------------------------------------

    virtual stdbool mouseButtonReceiver(const MouseButtonEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit)) =0;
    virtual stdbool mouseMoveReceiver(const MouseMoveEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit)) =0;

    //----------------------------------------------------------------
    //
    // Draw.
    //
    //----------------------------------------------------------------

    using DrawKit = GpuModuleProcessKit;

    struct DrawArgs
    {
        OverlayBuffer& overlay;
        LogBuffer& globalLog;
        LogBuffer& localLog;
        const GpuMatrix<uint8_x4>& dstImage;
    };

    virtual stdbool draw(const DrawArgs& args, stdPars(DrawKit)) =0;
};

//----------------------------------------------------------------

}
