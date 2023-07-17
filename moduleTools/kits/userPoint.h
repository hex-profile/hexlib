#pragma once

#include "numbers/float/floatBase.h"
#include "point/pointBase.h"

//================================================================
//
// UserPointKit
//
//================================================================

struct UserPoint
{
    bool valid = false;
    Point<float32> floatPos = point(0.f);

    bool leftSet = false;
    bool leftReset = false;
    bool rightSet = false;
    bool rightReset = false;
};
