#pragma once

#include "numbers/int/intBase.h"

//================================================================
//
// Basic random number generator.
//
// 32-bit internal state and functions for generating
// random 16-bit and 32-bit unsigned integers.
//
//================================================================

//================================================================
//
// RndgenState
//
//================================================================

using RndgenState = uint32;

//================================================================
//
// rndgenNext
//
// Should be 1 instruction if in series.
//
//================================================================

sysinline void rndgenNext(RndgenState& state)
{
    state = state * uint32(1664525UL) + uint32(1013904223UL);
}

//================================================================
//
// rndgen16
// rndgen32
//
//================================================================

sysinline uint32 rndgen16(RndgenState& state)
{
    uint32 s = state;
    rndgenNext(s);
    state = s;
    return (s >> 16);
}

//----------------------------------------------------------------

sysinline uint32 rndgenBetter16(RndgenState& state)
{
    uint32 x = state;
    x = ((x) ^ (x<<5) ^ (x>>27) ^ (x<<24) ^ (x>>8)) + 0x37798849;
    state = x;
    return (x >> 16);
}

//================================================================
//
// distributiveHash
//
// Thomas Wang hash, generates 9 instructions on Kepler.
//
//================================================================

sysinline uint32 distributiveHash(uint32 seed)
{
    seed = (seed ^ 61) ^ (seed >> 16); // 3
    seed *= 9; // 1
    seed = seed ^ (seed >> 4); // 2
    seed *= 0x27d4eb2d; // 1
    seed = seed ^ (seed >> 15); // 2

    return seed;
}

//================================================================
//
// rndgen32
//
//================================================================

sysinline uint32 rndgen32(RndgenState& state)
{
    return (uint32(rndgen16(state)) << 16) + uint32(rndgen16(state));
}

//================================================================
//
// rndgenBool
//
//================================================================

sysinline bool rndgenBool(RndgenState& state)
{
    return (rndgen16(state) & 0x8000) != 0;
}
