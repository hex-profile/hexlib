#pragma once

#include "lib/keys/keyBase.h"
#include "numbers/float/floatBase.h"
#include "point/pointBase.h"
#include "stdFunc/stdFunc.h"
#include "storage/adapters/callable.h"

//================================================================
//
// RefreshReceiver
//
// Receives 'window needs refresh' event.
//
//================================================================

using RefreshReceiver = Callable<void (stdNullPars)>;

KIT_CREATE(RefreshReceiverKit, RefreshReceiver&, refreshReceiver);

//================================================================
//
// KeyReceiver
//
//================================================================

using KeyReceiver = Callable<void (const KeyEvent& event, stdNullPars)>;

KIT_CREATE(KeyReceiverKit, KeyReceiver&, keyReceiver);

//================================================================
//
// MouseMoveReceiver
//
//================================================================

using MouseMoveEvent = Point<float32>;

using MouseMoveReceiver = Callable<void (const MouseMoveEvent& event, stdNullPars)>;

KIT_CREATE(MouseMoveReceiverKit, MouseMoveReceiver&, mouseMoveReceiver);

//================================================================
//
// MouseButtonReceiver
//
//================================================================

struct MouseButtonEvent
{
    Point<float32> position;
    int button;
    bool press;
    KeyModifiers modifiers;
};

////

using MouseButtonReceiver = Callable<void (const MouseButtonEvent& event, stdNullPars)>;

////

KIT_CREATE(MouseButtonReceiverKit, MouseButtonReceiver&, mouseButtonReceiver);

//================================================================
//
// ScrollReceiver
//
//================================================================

using ScrollEvent = Point<float32>;

using ScrollReceiver = Callable<void (const ScrollEvent& event, stdNullPars)>;

KIT_CREATE(ScrollReceiverKit, ScrollReceiver&, scrollReceiver);

//================================================================
//
// ResizeReceiver
//
//================================================================

using ResizeEvent = Point<Space>; // New framebuffer size in pixels

using ResizeReceiver = Callable<void (const ResizeEvent& event, stdNullPars)>;

KIT_CREATE(ResizeReceiverKit, ResizeReceiver&, resizeReceiver);

//================================================================
//
// EventReceivers
//
//================================================================

using EventReceivers = KitCombine
<
    RefreshReceiverKit,
    KeyReceiverKit,
    MouseMoveReceiverKit,
    MouseButtonReceiverKit,
    ScrollReceiverKit,
    ResizeReceiverKit
>;
