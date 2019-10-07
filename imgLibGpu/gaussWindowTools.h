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
    float32 divSigmaSq;
    Space taps;
};

//================================================================
//
// makeGaussWindowParams
//
//================================================================

GaussWindowParams makeGaussWindowParams(float32 sigma);

//================================================================
//
// GaussWindow
//
//================================================================

template <Space gaussQuality = 4>
class GaussWindowWeight
{

public:

    sysinline GaussWindowWeight(float32 sigma)
        {divSigmaSq = nativeRecipZero(square(sigma));}

    sysinline GaussWindowWeight(const GaussWindowParams& params)
        {divSigmaSq = params.divSigmaSq;}

    sysinline float32 weightDistSq(float32 distSq)
        {return gaussExpoApprox<gaussQuality>(distSq * divSigmaSq);}

    sysinline float32 weightDist(float32 delta)
        {return weightDistSq(square(delta));}

    sysinline float32 weightDist(const Point<float32>& delta)
        {return weightDistSq(square(delta.X) + square(delta.Y));}

    sysinline float32 weightDist(float32 dX, float32 dY)
        {return weightDistSq(square(dX) + square(dY));}

private:

    float32 divSigmaSq = 0;

};
