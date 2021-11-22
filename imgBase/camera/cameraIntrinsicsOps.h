#pragma once

#include "camera/cameraIntrinsics.h"
#include "point/pointMathIntrinsics.h"
#include "point3d/point3dBase.h"

//================================================================
//
// apply
//
//================================================================

template <typename Float>
sysinline Point<Float> operator %(const CameraIntrinsics<Float>& c, const Point<Float>& value)
{
    return value * c.focal + c.center;
}

//----------------------------------------------------------------

template <typename Float>
sysinline Point<Float> operator %(const CameraIntrinsics<Float>& c, const Point3D<Float>& value)
{
    auto ray = nativeRecipZero(value.Z) * point(value.X, value.Y);
    return c % ray;
}

//================================================================
//
// applyGetPoint
//
//================================================================

template <typename Float>
sysinline Point3D<Float> applyGetPoint(const CameraIntrinsics<Float>& intrinsicsInverse, const Point<Float>& screenPos, Float Z)
{
    auto ray = intrinsicsInverse % screenPos;
    return {Z * ray.X, Z * ray.Y, Z};
}

//================================================================
//
// inverse
//
//================================================================

template <typename Float>
sysinline auto operator ~(const CameraIntrinsics<Float>& c)
{
    auto invf = nativeRecipZero(c.focal);
    return CameraIntrinsics<Float>(invf, -invf * c.center);
}

//================================================================
//
// scale
//
// If resolution increases, factor > 1.
//
//================================================================

template <typename Float>
sysinline auto scale(const CameraIntrinsics<Float>& c, const Point<Float>& factor)
{
    return CameraIntrinsics<Float>(c.focal * factor, c.center * factor);
}

template <typename Float>
sysinline auto scale(const CameraIntrinsics<Float>& c, Float factor)
{
    return CameraIntrinsics<Float>(c.focal * factor, c.center * factor);
}

//================================================================
//
// importOpenCV
//
//================================================================

template <typename Float>
sysinline auto importOpenCV(const CameraIntrinsics<Float>& c)
{
    return CameraIntrinsics<Float>(c.focal, c.center + 0.5f);
}

//================================================================
//
// exportOpenCV
//
//================================================================

template <typename Float>
sysinline auto exportOpenCV(const CameraIntrinsics<Float>& c)
{
    return CameraIntrinsics<Float>(c.focal, c.center - 0.5f);
}

//================================================================
//
// Traits.
//
//================================================================

VECTOR_BASE_REBASE_VECTOR_IMPL(CameraIntrinsics)
TYPE_CONTROL_VECTOR_IMPL(CameraIntrinsics)

//================================================================
//
// Conversions.
//
//================================================================

CONVERT_FAMILY_VECTOR_IMPL(CameraIntrinsics, CameraIntrinsicsFamily)

//----------------------------------------------------------------

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<CameraIntrinsicsFamily, CameraIntrinsicsFamily, check, rounding, hint>
{
    template <typename SrcType, typename DstType>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcType);
        using DstBase = VECTOR_BASE(DstType);

        using BaseImpl = typename ConvertImplCall<Point<SrcBase>, Point<DstBase>, check, rounding, hint>::Code;

        static sysinline CameraIntrinsics<DstBase> func(const CameraIntrinsics<SrcBase>& src)
        {
            return CameraIntrinsics<DstBase>
            (
                BaseImpl::func(src.focal), 
                BaseImpl::func(src.center)
            );
        }
    };
};
