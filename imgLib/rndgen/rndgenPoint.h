#pragma once

#include "rndgen/rndgenFloat.h"
#include "compileTools/compileTools.h"
#include "point/pointBase.h"

//================================================================
//
// rndgenUniform<Point>
//
//================================================================

template <typename Type>
sysinline void rndgenUniform(RndgenState& state, Point<Type>& result)
{
    rndgenUniform(state, result.X);
    rndgenUniform(state, result.Y);
}

//================================================================
//
// rndgenPointGaussApprox3
// rndgenPointGaussApprox4
//
//================================================================

sysinline Point<float32> rndgenPointGaussApprox3(RndgenState& state)
{
    float32 rX = rndgenGaussApprox3(state);
    float32 rY = rndgenGaussApprox3(state);
    return point(rX, rY);
}

sysinline Point<float32> rndgenPointGaussApprox4(RndgenState& state)
{
    float32 rX = rndgenGaussApprox4(state);
    float32 rY = rndgenGaussApprox4(state);
    return point(rX, rY);
}
