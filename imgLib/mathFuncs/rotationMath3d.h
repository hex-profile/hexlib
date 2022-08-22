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
sysinline Point4D<Float> operator ~(const Point4D<Float>& Q)
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
sysinline Point4D<Float> operator %(const Point4D<Float>& A, const Point4D<Float>& B)
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
// quatApply
//
// The quaternion should have unit length.
//
// 18 FMADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> operator %(const Point4D<Float>& Q, const Point3D<Float>& V)
{
    auto R = quatImaginary(Q);
	return V + 2 * crossProduct(R, crossProduct(R, V) + quatReal(Q) * V);
}

//================================================================
//
// Mat3D
//
// For efficient target application of a rotation.
//
//================================================================

template <typename Float>
using Mat3D = Point3D<Point3D<Float>>;

//================================================================
//
// quatMat
//
//================================================================

template <typename Float>
sysinline Mat3D<Float> quatMat(const Point4D<Float>& Q)
{
    auto XX = 2 * Q.X * Q.X;
    auto XY = 2 * Q.X * Q.Y;
    auto XZ = 2 * Q.X * Q.Z;
    auto XW = 2 * Q.X * Q.W;

    auto YY = 2 * Q.Y * Q.Y;
    auto YZ = 2 * Q.Y * Q.Z;
    auto YW = 2 * Q.Y * Q.W;

    auto ZZ = 2 * Q.Z * Q.Z;
    auto ZW = 2 * Q.Z * Q.W;

    return point3D
    (
        point3D(1.f - YY - ZZ, XY - ZW, YW + XZ), 
        point3D(ZW + XY, 1.f - ZZ - XX, YZ - XW), 
        point3D(XZ - YW, XW + YZ, 1.f - XX - YY)
    );
}

//================================================================
//
// Mat3D inverse.
//
//================================================================

template <typename Float>
sysinline Mat3D<Float> operator ~(const Mat3D<Float>& R)
{
    return point3D
    (
        point3D(R.X.X, R.Y.X, R.Z.X),
        point3D(R.X.Y, R.Y.Y, R.Z.Y),
        point3D(R.X.Z, R.Y.Z, R.Z.Z)
    );
}

//================================================================
//
// Mat3D apply.
//
// 9 MADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> operator %(const Mat3D<Float>& R, const Point3D<Float>& V)
{
    return point3D
    (
        scalarProd(R.X, V),
        scalarProd(R.Y, V),
        scalarProd(R.Z, V)
    );
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
    fastCosSin(vecLength, rX, rY);

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

    Float theta2 = exactAtan2(rY, rX);

    return theta2 * vecDir;
}

//================================================================
//
// quatFromRodrigues
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatFromRodrigues(const Point3D<Float>& R)
{
    return quatImaginaryExp(0.5f * R);
}

//================================================================
//
// quatToRodrigues
//
// The quaternion should have unit length.
// The function takes the shortest Rodrigues path.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatToRodrigues(const Point4D<Float>& Q)
{
    return 2 * quatUnitLogSpecial(Q);
}

//================================================================
//
// quatFlipToBase
//
// Flips a quaternion sign to make it closer to the specified base 
// as if they were R^4 vectors.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatFlipToBase(const Point4D<Float>& Q, const Point4D<Float>& base)
{
    Float lenSq1 = vectorLengthSq(Q - base);
    Float lenSq2 = vectorLengthSq(Q + base);

    return (lenSq1 < lenSq2) ? +Q : -Q;
}
