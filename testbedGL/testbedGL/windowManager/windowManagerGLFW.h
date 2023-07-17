#pragma once

#include "allocation/mallocKit.h"
#include "data/array.h"
#include "errorLog/errorLogKit.h"
#include "lib/eventReceivers.h"
#include "lib/keys/keyBase.h"
#include "point/point.h"
#include "stdFunc/stdFunc.h"
#include "storage/adapters/lambdaThunk.h"
#include "storage/opaqueStruct.h"
#include "storage/smartPtr.h"
#include "userOutput/msgLogExKit.h"
#include "storage/optionalObject.h"
#include "lib/contextBinder.h"

//================================================================
//
// WindowMode
//
//================================================================

enum class WindowMode {Minimized, Normal, Maximized, FullScreen, COUNT};

//================================================================
//
// WindowLocation
//
//================================================================

struct WindowLocation
{
    WindowMode mode = WindowMode::Normal;
    bool decorated = false;
    bool verticalSync = false;
    Point<Space> pos = point(0);
    Point<Space> size = point(0);
};

//================================================================
//
// WindowCreationArgs
//
//================================================================

struct WindowCreationArgs
{
    const char* name = nullptr;
    WindowLocation location;
    bool resizable = true;
};

//================================================================
//
// Window
//
//================================================================

struct Window
{
    using Kit = KitCombine<ErrorLogKit, MsgLogExKit>;

    virtual ~Window() {}

    virtual stdbool setThreadDrawingContext(stdPars(Kit)) =0;

    virtual stdbool setVisible(bool visible, stdPars(Kit)) =0;

    virtual stdbool getEvents(bool waitEvents, const OptionalObject<uint32>& waitTimeoutMs, const EventReceivers& receivers, stdPars(Kit)) =0;

    virtual stdbool shouldContinue(stdPars(Kit)) =0;

    virtual stdbool swapBuffers(stdPars(Kit)) =0;

    virtual stdbool getWindowLocation(WindowLocation& location, stdPars(Kit)) =0;
    virtual stdbool setWindowLocation(const WindowLocation& location, stdPars(Kit)) =0;

    virtual stdbool getImageSize(Point<Space>& size, stdPars(Kit)) =0;
};

//================================================================
//
// WindowManager
//
//================================================================

struct WindowManager
{
    using Kit = KitCombine<ErrorLogKit, MsgLogExKit, MallocKit>;

    virtual void postEmptyEvent() =0; // can be called from a concurrent thread

    virtual stdbool getCurrentDisplayResolution(Point<Space>& result, stdPars(Kit)) =0;

    virtual stdbool createWindow(UniquePtr<Window>& window, const WindowCreationArgs& par, stdPars(Kit)) =0;

    virtual stdbool createOffscreenGLContext(UniquePtr<ContextBinder>& context, stdPars(Kit)) =0;
};

//================================================================
//
// WindowManagerGLFW
//
//================================================================

class WindowManagerGLFW : public WindowManager
{

public:

    stdbool init(stdPars(Kit));
    void deinit();

    inline ~WindowManagerGLFW() {deinit();}

public:

    void postEmptyEvent(); // can be called from a concurrent thread

    stdbool getCurrentDisplayResolution(Point<Space>& result, stdPars(Kit));

    stdbool createWindow(UniquePtr<Window>& window, const WindowCreationArgs& par, stdPars(Kit));

    stdbool createOffscreenGLContext(UniquePtr<ContextBinder>& context, stdPars(Kit));

private:

    bool initialized = false;

};
