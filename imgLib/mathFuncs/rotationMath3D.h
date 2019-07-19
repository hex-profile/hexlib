#pragma once

#include "point3d/point3d.h"
#include "point4d/point4d.h"
#include "numbers/mathIntrinsics.h"
#include "mathFuncs/quatBase.h"

//================================================================
//
// quatConjugate
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatConjugate(const Point4D<Float>& Q)
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
// The quaternion should have unit length.
//
// 18 FMADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatRotateVec(const Point4D<Float>& Q, const Point3D<Float>& V)
{
    auto R = quatImaginary(Q);
	return V + 2 * crossProduct(R, crossProduct(R, V) + quatReal(Q) * V);
}

//================================================================
//
// quatImaginaryExp
//
// For rotation:
//   * The vector direction is the rotation axis.
//   * The vector length is (theta/2) in radians clockwise.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatImaginaryExp(const Point3D<Float>& vec)
{
    VECTOR_DECOMPOSE(vec);

    Float rX, rY;
    nativeCosSin(vecLength, rX, rY);

    return quatCompose(rX, rY * vecDir);
}

//================================================================
//
// quatUnitLogSpecial
//
//----------------------------------------------------------------
//
// The quaternion is {cos(theta/2), sin(theta/2) * dir}.
// Function needs to return (theta/2) * dir.
//
// The function returns (theta/2) in range [-pi/2, +pi/2] 
// to minimize the angle magnitude.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatUnitLogSpecial(const Point4D<Float>& Q)
{
    auto vec = quatImaginary(Q);
    Float rX = quatReal(Q);

    //
    // To minimize angle magnitude, use (rX > 0) and (rY > 0).
    //
    // If rX < 0, invert the input quaternion, 
    // which does not change its SO(3) rotation action.
    //

    if (rX < 0)
        {rX = -rX; vec = -vec;}

    ////

    VECTOR_DECOMPOSE(vec);
    Float rY = vecLength; // >= 0

    ////

    Float theta2 = nativeAtan2(rY, rX);

    return theta2 * vecDir;
}

//================================================================
//
// quatBoxPlus
//
// The quaternion should have unit length.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatBoxPlus(const Point4D<Float>& Q, const Point3D<Float>& D)
{
    return quatMul(Q, quatImaginaryExp(0.5f * D));
}

//================================================================
//
// quatBoxMinus
//
// The quaternions should have unit length.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatBoxMinus(const Point4D<Float>& Q, const Point4D<Float>& P)
{
    return 2 * quatUnitLogSpecial(quatMul(quatConjugate(P), Q));
}

//================================================================
//
// quatL2Diff
//
//================================================================

template <typename Float>
Float quatL2Diff(const Point4D<Float>& A, const Point4D<Float>& B)
{
    Float lenSq1 = vectorLengthSq(A - B);
    Float lenSq2 = vectorLengthSq(A + B);
    return fastSqrt(minv(lenSq1, lenSq2));
}
