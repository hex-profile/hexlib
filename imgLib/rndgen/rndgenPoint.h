#pragma once

#include "rndgen/rndgenFloat.h"
#include "compileTools/compileTools.h"
#include "point/pointBase.h"
#include "point3d/point3dBase.h"
#include "point4d/point4dBase.h"

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
// rndgenGaussApproxFour<Point<float32>>
//
//================================================================

template <>
sysinline Point<float32> rndgenGaussApproxFour(RndgenState& state)
{
    auto rX = rndgenGaussApproxFour<float32>(state);
    auto rY = rndgenGaussApproxFour<float32>(state);
    return point(rX, rY);
}

template <>
sysinline Point3D<float32> rndgenGaussApproxFour(RndgenState& state)
{
    auto rX = rndgenGaussApproxFour<float32>(state);
    auto rY = rndgenGaussApproxFour<float32>(state);
    auto rZ = rndgenGaussApproxFour<float32>(state);
    return point3D(rX, rY, rZ);
}

template <>
sysinline Point4D<float32> rndgenGaussApproxFour(RndgenState& state)
{
    auto rX = rndgenGaussApproxFour<float32>(state);
    auto rY = rndgenGaussApproxFour<float32>(state);
    auto rZ = rndgenGaussApproxFour<float32>(state);
    auto rW = rndgenGaussApproxFour<float32>(state);
    return point4D(rX, rY, rZ, rW);
}
