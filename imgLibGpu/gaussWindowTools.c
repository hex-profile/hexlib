#include "gaussWindowTools.h"

//================================================================
//
// maxSigmaForSlidingGaussWith8bitAccuracy
//
// Maximal sigma for the given number of taps 
// to get (255/256) Gauss coverage.
//
//================================================================

static const float32 maxSigmaForSlidingGaussWith8bitAccuracy[] =
{
    -1, 
    0.012f, // 1 taps
    0.300f, // 2 taps
    0.454f, // 3 taps
    0.626f, // 4 taps
    0.801f, // 5 taps
    0.978f, // 6 taps
    1.156f, // 7 taps
    1.334f, // 8 taps
    1.511f, // 9 taps
    1.688f, // 10 taps
    1.865f, // 11 taps
    2.041f, // 12 taps
    2.217f, // 13 taps
    2.393f, // 14 taps
    2.568f, // 15 taps
    2.743f, // 16 taps
};

//================================================================
//
// makeGaussWindowParams
//
//================================================================

GaussWindowParams makeGaussWindowParams(float32 sigma)
{
    float32 coverageFactor = 3.0f;
    Space taps = clampMin(convertUp<Space>(2 * coverageFactor * sigma), 1);

    ////

    for_count (i, Space(COMPILE_ARRAY_SIZE(maxSigmaForSlidingGaussWith8bitAccuracy)))
    {
        if (sigma <= maxSigmaForSlidingGaussWith8bitAccuracy[i])
            {taps = i; break;}
    }

    ////

    GaussWindowParams params;
    params.divSigmaSq = nativeRecipZero(square(sigma));
    params.taps = taps;

    return params;
}
