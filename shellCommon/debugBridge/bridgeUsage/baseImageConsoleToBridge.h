#pragma once

#include "baseInterfaces/baseImageConsole.h"
#include "debugBridge/bridge/debugBridge.h"
#include "kits/msgLogsKit.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogExKit.h"

namespace baseImageConsoleToBridge {

//================================================================
//
// Kit
//
//================================================================

using Kit = KitCombine<LocalLogKit, ErrorLogKit, MsgLogExKit, MessageFormatterKit>;

//================================================================
//
// BaseVideoOverlayToBridge
//
//================================================================

class BaseVideoOverlayToBridge : public BaseVideoOverlay
{

public:

    BaseVideoOverlayToBridge(debugBridge::VideoOverlay& destOverlay, const Kit& kit)
        : destOverlay{destOverlay}, kit{kit} {}

public:

    stdbool overlayClear(stdParsNull);
    stdbool overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull);
    stdbool overlaySetFake(stdParsNull);
    stdbool overlayUpdate(stdParsNull);

private:

    debugBridge::VideoOverlay& destOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using baseImageConsoleToBridge::BaseVideoOverlayToBridge;
