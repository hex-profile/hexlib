#pragma once

#include "rndgen/rndgenBase.h"

//================================================================
//
// rndgenMix*
//
// Use with caution. Distribution properties have not been verified. For example,
// when mixing X and Y in 2D, with just two iterations major problems
// are apparent. With three iterations it seems okay.
//
//================================================================

sysinline void rndgenMix3(RndgenState& rndgen, RndgenState value)
{
    rndgen ^= value;
    rndgenNext(rndgen);

    rndgen ^= value;
    rndgenNext(rndgen);

    rndgen ^= value;
    rndgenNext(rndgen);
}

////

sysinline void rndgenMix4(RndgenState& rndgen, RndgenState value)
{
    rndgen ^= value;
    rndgenNext(rndgen);

    rndgen ^= value;
    rndgenNext(rndgen);

    rndgen ^= value;
    rndgenNext(rndgen);

    rndgen ^= value;
    rndgenNext(rndgen);
}
