#pragma once

#include "kit/kit.h"
#include "kits/userPoint.h"

//================================================================
//
// AtVideoFrame
//
//================================================================

struct AtVideoFrame;

KIT_CREATE(AtVideoFrameKit, const AtVideoFrame&, atVideoFrame);

//================================================================
//
// AtAsyncOverlay
//
// THE INTERFACE POINTER DOES NOT CHANGE AT RUNTIME
//
//================================================================

struct AtAsyncOverlay;

KIT_CREATE(AtAsyncOverlayKit, AtAsyncOverlay&, atAsyncOverlay);

//================================================================
//
// AtVideoOverlayKit
//
//================================================================

using BaseVideoOverlay = struct BaseVideoOverlay;

KIT_CREATE(AtVideoOverlayKit, BaseVideoOverlay&, atVideoOverlay);

//================================================================
//
// AtOutImgKit
//
//================================================================

using BaseImageConsole = struct BaseImageConsole;

KIT_CREATE(AtImgConsoleKit, BaseImageConsole&, atImgConsole);

//================================================================
//
// BaseActionSetup
//
//================================================================

struct BaseActionSetup;

KIT_CREATE(AtSignalSetKit, BaseActionSetup&, atSignalSet);

//================================================================
//
// BaseActionReceiving
//
//================================================================

struct BaseActionReceiving;

KIT_CREATE(AtSignalTestKit, BaseActionReceiving&, atSignalTest);

//================================================================
//
// AtVideoInfo
//
//================================================================

struct AtVideoInfo;

KIT_CREATE(AtVideoInfoKit, const AtVideoInfo&, atVideoInfo);

//================================================================
//
// AtUserPointKit
//
//================================================================

KIT_CREATE2(AtUserPointKit, bool, atUserPointValid, Point<Space>, atUserPoint);

//================================================================
//
// AtContinousModeKit
//
//================================================================

KIT_CREATE2(AtContinousModeKit, bool, atRunning, bool, atPlaying);

//================================================================
//
// AtSetBusyStatusKit
//
//================================================================

struct AtSetBusyStatus;

KIT_CREATE(AtSetBusyStatusKit, AtSetBusyStatus&, atSetBusyStatus);

//================================================================
//
// AtCommonKit
// AtProcessKit
//
//================================================================

using AtCommonKit = KitCombine<AtImgConsoleKit, AtSignalSetKit>;

using AtProcessKit = KitCombine<AtCommonKit, AtVideoFrameKit, AtVideoOverlayKit, AtAsyncOverlayKit, 
    AtSignalTestKit, AtVideoInfoKit, AtUserPointKit, AtContinousModeKit>;

