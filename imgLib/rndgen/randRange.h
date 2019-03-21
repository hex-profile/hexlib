#pragma once

#include "rndgen/rndgenBase.h"
#include "numbers/int/intBase.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// rand
//
//================================================================

template <typename Type>
Type rand(RndgenState& rndgen, Type lo, Type hi);

template <>
uint32 rand(RndgenState& rndgen, uint32 lo, uint32 hi);

template <>
int32 rand(RndgenState& rndgen, int32 lo, int32 hi);

template <>
float32 rand(RndgenState& rndgen, float32 lo, float32 hi);

template <>
float64 rand(RndgenState& rndgen, float64 lo, float64 hi);
