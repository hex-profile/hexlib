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
    bool valid = false; 
    Point<Space> position = point(0);
    bool leftSet = false;
    bool leftReset = false;
    bool rightSet = false;
    bool rightReset = false;

    UserPoint() =default;

    UserPoint(bool valid, const Point<Space>& position, bool leftSet, bool leftReset, bool rightSet, bool rightReset)
        : valid(valid), position(position), leftSet(leftSet), leftReset(leftReset), rightSet(rightSet), rightReset(rightReset)
    {
    }
};

KIT_CREATE(UserPointKit, const UserPoint&, userPoint);
