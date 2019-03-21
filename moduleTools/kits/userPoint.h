#pragma once

#include "point/point.h"
#include "data/space.h"
#include "kit/kit.h"

//================================================================
//
// UserPointKit
//
//================================================================

KIT_CREATE4(UserPoint, bool, valid, Point<Space>, position, bool, signal, bool, signalAlt);

KIT_CREATE1(UserPointKit, const UserPoint&, userPoint);
