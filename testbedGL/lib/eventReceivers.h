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

using RefreshReceiver = Callable<stdbool (stdParsNull)>;

KIT_CREATE(RefreshReceiverKit, RefreshReceiver&, refreshReceiver);

//================================================================
//
// KeyReceiver
//
//================================================================

using KeyReceiver = Callable<stdbool (const KeyEvent& event, stdParsNull)>;

KIT_CREATE(KeyReceiverKit, KeyReceiver&, keyReceiver);

//================================================================
//
// MouseMoveReceiver
//
//================================================================

using MouseMoveEvent = Point<float32>;

using MouseMoveReceiver = Callable<stdbool (const MouseMoveEvent& event, stdParsNull)>;

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

using MouseButtonReceiver = Callable<stdbool (const MouseButtonEvent& event, stdParsNull)>;

////

KIT_CREATE(MouseButtonReceiverKit, MouseButtonReceiver&, mouseButtonReceiver);

//================================================================
//
// ScrollReceiver
//
//================================================================

using ScrollEvent = Point<float32>;

using ScrollReceiver = Callable<stdbool (const ScrollEvent& event, stdParsNull)>;

KIT_CREATE(ScrollReceiverKit, ScrollReceiver&, scrollReceiver);

//================================================================
//
// ResizeReceiver
//
//================================================================

using ResizeEvent = Point<Space>; // New framebuffer size in pixels

using ResizeReceiver = Callable<stdbool (const ResizeEvent& event, stdParsNull)>;

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
