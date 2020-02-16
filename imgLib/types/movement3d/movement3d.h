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
sysinline bool operator ==(const Movement3D<Float>& A, const Movement3D<Float>& B)
{
    return allv(A.rotation == B.rotation) && allv(A.translation == B.translation);
}

//================================================================
//
// apply
//
//================================================================

template <typename Float>
sysinline Point3D<Float> operator %(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return movement.rotation % vec + movement.translation;
}

//================================================================
//
// combine
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> operator %(const Movement3D<Float>& B, const Movement3D<Float>& A)
{
    Movement3D<Float> result;
    result.rotation = B.rotation % A.rotation;
    result.translation = B.rotation % A.translation + B.translation;
    return result;
}

//================================================================
//
// inverse
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> operator ~(const Movement3D<Float>& movement)
{
    Movement3D<Float> result;
    auto iR = ~movement.rotation;
    result.rotation = iR;
    result.translation = -(iR % movement.translation);
    return result;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Point3D<Float> applyInverse(const Movement3D<Float>& movement, const Point3D<Float>& vec)
{
    return ~movement.rotation % (vec - movement.translation);
}

//================================================================
//
// normalize
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> normalize(const Movement3D<Float>& movement)
{
    return movement3D(vectorNormalize(movement.rotation), movement.translation);
}

//================================================================
//
// MovementUnpacked3D
//
//================================================================

template <typename Float>
struct MovementUnpacked3D
{

public:

    Mat3D<Float> rotation{};
    Point3D<Float> translation{};

public:

    sysinline MovementUnpacked3D() =default;

    sysinline MovementUnpacked3D(const Movement3D<Float>& base)
    {
        rotation = quatMat(base.rotation);
        translation = base.translation;
    }
};

////

template <typename Float>
sysinline auto movementUnpacked3D(const Movement3D<Float>& movement)
    {return MovementUnpacked3D<Float>(movement);}

//================================================================
//
// apply
//
//================================================================

template <typename Float>
sysinline Point3D<Float> operator %(const MovementUnpacked3D<Float>& movement, const Point3D<Float>& vec)
{
    return movement.rotation % vec + movement.translation;
}

//================================================================
//
// importMovementWithRotationBase
//
// Input movement:
// (1) Rotate it around the given center by the given rotation.
// (2) Translate it by the given translation.
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> importMovementWithRotationBase(const Movement3D<Float>& movement, const Point3D<Float>& center)
{
    return movement3D
    (
        movement.rotation,
        movement.translation + center - (movement.rotation % center)
    );
}

//================================================================
//
// exportMovementWithRotationBase
//
//================================================================

template <typename Float>
sysinline Movement3D<Float> exportMovementWithRotationBase(const Movement3D<Float>& movement, const Point3D<Float>& center)
{
    return movement3D
    (
        movement.rotation,
        movement.translation + (movement.rotation % center) - center
    );
}

//================================================================
//
// Traits.
//
//================================================================

VECTOR_BASE_REBASE_VECTOR_IMPL(Movement3D)
TYPE_CONTROL_VECTOR_IMPL(Movement3D)

//================================================================
//
// Conversions.
//
//================================================================

CONVERT_FAMILY_VECTOR_IMPL(Movement3D, Movement3DFamily)

//----------------------------------------------------------------

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<Movement3DFamily, Movement3DFamily, check, rounding, hint>
{
    template <typename SrcType, typename DstType>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcType);
        using DstBase = VECTOR_BASE(DstType);

        using BaseImpl4D = typename ConvertImplCall<Point4D<SrcBase>, Point4D<DstBase>, check, rounding, hint>::Code;
        using BaseImpl3D = typename ConvertImplCall<Point3D<SrcBase>, Point3D<DstBase>, check, rounding, hint>::Code;

        static sysinline Movement3D<DstBase> func(const Movement3D<SrcBase>& src)
        {
            return movement3D
            (
                BaseImpl4D::func(src.rotation), 
                BaseImpl3D::func(src.translation)
            );
        }
    };
};

//================================================================
//
// fullyDefined
//
//================================================================

template <typename Float>
inline bool fullyDefined(const Movement3D<Float>& movement)
{
    return allv(def(movement.rotation) && allv(def(movement.translation)));
}
