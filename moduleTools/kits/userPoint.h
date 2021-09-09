#pragma once

#include "point/point.h"
#include "data/space.h"
#include "kit/kit.h"

//================================================================
//
// UserPointKit
//
//================================================================

struct UserPoint
{
    bool valid; 
    Point<Space> position; 
    bool signal; 
    bool signalAlt;
};

KIT_CREATE(UserPointKit, const UserPoint&, userPoint);
