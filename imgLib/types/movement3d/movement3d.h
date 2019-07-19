#pragma once

#include "movement3dBase.h"
#include "mathFuncs/rotationMath3d.h"

//================================================================
//
// apply
//
//================================================================

template <typename Float>
sysinline Point3D<Float> apply(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return quatRotateVec(movement.rotation, vec) + movement.translation;
}

//================================================================
//
// combine
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> combine(const Movement3D<Float>& A, const Movement3D<Float>& B)
{
    Movement3D<Float> result;
    result.rotation = quatMul(B.rotation, A.rotation);
    result.translation = quatRotateVec(B.rotation, A.translation) + B.translation;
    return result;
}

//================================================================
//
// inverse
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> inverse(const Movement3D<Float>& movement)
{
    Movement3D<Float> result;
    auto oppositeRotation = quatConjugate(movement.rotation);
    result.rotation = oppositeRotation;
    result.translation = -quatRotateVec(oppositeRotation, movement.translation);
    return result;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Point3D<Float> applyInverse(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return quatRotateVec(quatConjugate(movement.rotation), vec - movement.translation);
}
