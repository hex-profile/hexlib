#pragma once

#include "rndgen/rndgenBase.h"
#include "numbers/int/intBase.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// randRange
//
//================================================================

template <typename Type>
Type randRange(RndgenState& rndgen, Type lo, Type hi);
