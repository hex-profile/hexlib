#pragma once

#include "kit/kit.h"
#include "kits/userPoint.h"

//================================================================
//
// AtVideoFrame
//
//================================================================

struct AtVideoFrame;

KIT_CREATE1(AtVideoFrameKit, const AtVideoFrame&, atVideoFrame);

//================================================================
//
// AtAsyncOverlay
//
// THE INTERFACE POINTER DOES NOT CHANGE AT RUNTIME
//
//================================================================

struct AtAsyncOverlay;

KIT_CREATE1(AtAsyncOverlayKit, AtAsyncOverlay&, atAsyncOverlay);

//================================================================
//
// AtVideoOverlayKit
//
//================================================================

using BaseVideoOverlay = struct BaseVideoOverlay;

KIT_CREATE1(AtVideoOverlayKit, BaseVideoOverlay&, atVideoOverlay);

//================================================================
//
// AtOutImgKit
//
//================================================================

using BaseImageConsole = struct BaseImageConsole;

KIT_CREATE1(AtImgConsoleKit, BaseImageConsole&, atImgConsole);

//================================================================
//
// BaseActionSetup
//
//================================================================

struct BaseActionSetup;

KIT_CREATE1(AtSignalSetKit, BaseActionSetup&, atSignalSet);

//================================================================
//
// BaseActionReceiving
//
//================================================================

struct BaseActionReceiving;

KIT_CREATE1(AtSignalTestKit, BaseActionReceiving&, atSignalTest);

//================================================================
//
// AtVideoInfo
//
//================================================================

struct AtVideoInfo;

KIT_CREATE1(AtVideoInfoKit, const AtVideoInfo&, atVideoInfo);

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

KIT_CREATE1(AtSetBusyStatusKit, AtSetBusyStatus&, atSetBusyStatus);

//================================================================
//
// AtCommonKit
// AtProcessKit
//
//================================================================

KIT_COMBINE2(AtCommonKit, AtImgConsoleKit, AtSignalSetKit);

KIT_COMBINE8(AtProcessKit, AtCommonKit, AtVideoFrameKit, AtVideoOverlayKit, AtAsyncOverlayKit, AtSignalTestKit, 
    AtVideoInfoKit, AtUserPointKit, AtContinousModeKit);

