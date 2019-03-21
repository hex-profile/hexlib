#pragma once

#include "mathFuncs/gaussApprox.h"
#include "data/space.h"
#include "point/point.h"
#include "imageRead/positionTools.h"

//================================================================
//
// GaussWindowParams
//
//================================================================

struct GaussWindowParams
{
    float32 normFactor;
    float32 divSigmaSq;
    Space taps;
};

//================================================================
//
// setupGaussWindowParams
//
//================================================================

sysinline void setupGaussWindowParams(float32 sigma, GaussWindowParams& params)
{
    float32 divSigma = nativeRecipZero(sigma);
    params.divSigmaSq = square(divSigma);
    params.normFactor = 0.3989422802f * divSigma;

    float32 coverageFactor = 3.33f;
    float32 filterRadius = coverageFactor * sigma;
    params.taps = clampMin(convertUp<Space>(2 * filterRadius), 1);
}

//----------------------------------------------------------------

sysinline GaussWindowParams makeGaussWindowParams(float32 sigma)
{
    GaussWindowParams params;
    setupGaussWindowParams(sigma, params);
    return params;
}

//================================================================
//
// GaussWindow
//
//================================================================

template <bool normalized, Space gaussQuality = 4>
class GaussWindowWeight
{

public:

    sysinline GaussWindowWeight(const GaussWindowParams& params)
        : params(params) {}

    sysinline float32 weightDistSq(float32 dist2)
    {
        float32 result = gaussExpoApprox<gaussQuality>(dist2 * params.divSigmaSq);
        if (normalized) result *= params.normFactor;
        return result;
    }

    sysinline float32 weightDist(float32 delta)
        {return weightDistSq(square(delta));}

    sysinline float32 weightDist(const Point<float32>& delta)
        {return weightDistSq(square(delta.X) + square(delta.Y));}

    sysinline float32 weightDist(float32 dX, float32 dY)
        {return weightDistSq(square(dX) + square(dY));}

private:

    GaussWindowParams params;

};
