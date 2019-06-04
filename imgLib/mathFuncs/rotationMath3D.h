#pragma once

#include "point3d/point3d.h"
#include "point4d/point4dBase.h"
#include "compileTools/compileTools.h"

//================================================================
//
// conjuugate
//
//================================================================

template <typename Float>
sysinline Point4D<Float> conjugate(const Point4D<Float>& Q)
{
    return {-Q.X, -Q.Y, -Q.Z, Q.W};
}

//================================================================
//
// quatMul
//
// 16 FMADs.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatMul(const Point4D<Float>& A, const Point4D<Float>& B)
{
    return
    {
	    A.W * B.X + A.X * B.W + A.Y * B.Z - A.Z * B.Y,
	    A.W * B.Y + A.Y * B.W + A.Z * B.X - A.X * B.Z,
	    A.W * B.Z + A.Z * B.W + A.X * B.Y - A.Y * B.X,
	    A.W * B.W - A.X * B.X - A.Y * B.Y - A.Z * B.Z
    };
}

//================================================================
//
// crossProduct
//
// 6 FMADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> crossProduct(const Point3D<Float>& A, const Point3D<Float>& B)
{
	return 
    {
		A.Y * B.Z - B.Y * A.Z,
		A.Z * B.X - B.Z * A.X,
		A.X * B.Y - B.X * A.Y
    };
}

//================================================================
//
// quatRotateVec
//
// The quaternion should be normalized.
//
// 18 FMADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatRotateVec(const Point4D<Float>& Q, const Point3D<Float>& V)
{
    auto R = Point3D<Float>{Q.X, Q.Y, Q.Z};
	auto UV = crossProduct(R, V);
	auto UUV = crossProduct(R, UV);

	return V + 2 * (Q.W * UV + UUV);
}
