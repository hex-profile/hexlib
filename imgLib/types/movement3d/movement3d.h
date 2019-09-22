#pragma once

#include "movement3dBase.h"
#include "mathFuncs/rotationMath3d.h"

//================================================================
//
// movement3D
//
// * Zero movement.
// * Only rotation.
// * Only translation.
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> movement3D()
{
    return movement3D(quatIdentity<Float>(), point3D<Float>(0));
}

//----------------------------------------------------------------

template <typename Float>
sysinline Movement3D<Float> movement3D(const Point4D<Float>& rotation)
{
    return movement3D(rotation, point3D<Float>(0));
}

//----------------------------------------------------------------

template <typename Float>
sysinline Movement3D<Float> movement3D(const Point3D<Float>& translation)
{
    return movement3D(quatIdentity<Float>(), translation);
}

//================================================================
//
// operator ==
//
//================================================================
 
template <typename Float>
sysinline bool operator ==(const Movement3D<Float>& a, const Movement3D<Float>& b)
{
    return allv(a.rotation == b.rotation) && allv(a.translation == b.translation);
}

//================================================================
//
// apply
//
//================================================================

template <typename Float>
sysinline Point3D<Float> apply(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return movement.rotation % vec + movement.translation;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Point3D<Float> operator %(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return apply(movement, vec);
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
    result.rotation = B.rotation % A.rotation;
    result.translation = B.rotation % A.translation + B.translation;
    return result;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Movement3D<Float> operator %(const Movement3D<Float>& A, const Movement3D<Float>& B)
{
    return combine(B, A);
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
    auto iR = ~movement.rotation;
    result.rotation = iR;
    result.translation = -(iR % movement.translation);
    return result;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Movement3D<Float> operator ~(const Movement3D<Float>& movement)
{
    return inverse(movement);
}

//----------------------------------------------------------------

template <typename Float>
sysinline Point3D<Float> applyInverse(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return ~movement.rotation % (vec - movement.translation);
}
