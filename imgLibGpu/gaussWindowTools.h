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
// GaussSigmaSquare
//
//================================================================

struct GaussSigmaSquare
{
    float32 sigmaSq;
};

//================================================================
//
// GaussWindow
//
//================================================================

template <Space gaussQuality = 4>
class GaussWindow
{

public:

    sysinline GaussWindow(float32 sigma) // ``` eradicate
        {divSigmaSq = nativeRecipZero(square(sigma));}

    sysinline GaussWindow(const GaussSigmaSquare& that)
        {divSigmaSq = nativeRecipZero(that.sigmaSq);}

    sysinline GaussWindow(const GaussWindowParams& params)
        {divSigmaSq = params.divSigmaSq;}

public:

    sysinline float32 byDistSq(float32 distSq) const
        {return gaussExpoApprox<gaussQuality>(distSq * divSigmaSq);}

public:

    template <typename Vector>
    sysinline float32 byDist(const Vector& vec) const
        {return byDistSq(vectorLengthSq(vec));}

    sysinline float32 byDist(float32 dX, float32 dY) const
        {return byDistSq(square(dX) + square(dY));}

private:

    float32 divSigmaSq = 0;

};
