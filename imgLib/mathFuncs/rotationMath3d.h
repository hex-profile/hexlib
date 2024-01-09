#pragma once

#include "point3d/point3d.h"
#include "point4d/point4d.h"
#include "numbers/mathIntrinsics.h"
#include "mathFuncs/quatBase.h"
#include "mathFuncs/rotationMath.h"

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
// quatMat
//
//================================================================

template <typename Float>
sysinline auto quatMat(const Point4D<Float>& Q)
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
// quatFromRodriguesInCircles
//
// Converts Rodrigues vector into a quaternion. Unlike the standard
// conversion function, this function accepts rotation amount in
// circles instead of radians, i.e., the rotation range is
// from -1/2 to 1/2, instead of from -pi to pi.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatFromRodriguesInCircles(const Point3D<Float>& R)
{
    return quatImaginaryExp(Float(0.5 * 2 * pi64) * R);
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

//================================================================
//
// hqCrossProductOfUnitVectors
//
// Computes cross product of vectors A and B.
//
// The function aims to produce a more precise orthogonal
// vector as a result.
//
// The inputs should have unit length!
//
//================================================================

template <typename Float>
sysinline Point3D<Float> hqCrossProductOfUnitVectors(const Point3D<Float>& A, const Point3D<Float>& B)
{
    auto result = crossProduct(A, B);

    result -= A * scalarProd(result, A);
    result -= B * scalarProd(result, B);

    return result;
}

//================================================================
//
// computeRotationQuaternion
//
// This function computes the quaternion that represents the rotation
// which aligns vector A with vector B.
//
// Checked by quatGenTest.
//
//================================================================

template <typename Float>
sysinline auto computeRotationQuaternion(const Point3D<Float>& Av, const Point3D<Float>& Bv)
{
    // normalize vectors
    auto A = vectorNormalize(Av);
    auto B = vectorNormalize(Bv);

    // rotation axis
    auto rotationAxis = vectorNormalize(hqCrossProductOfUnitVectors(A, B));

    // transition to the coordinate system in the plane of rotation
    auto xAxis = B;
    auto xCoord = scalarProd(A, xAxis);

    // Y axis of the rotation plane
    auto yAxis = vectorNormalize(hqCrossProductOfUnitVectors(rotationAxis, B));
    auto yCoord = scalarProd(A, yAxis);

    // consider X and Y coordinates as a complex number and take its phase
    auto rotationAmount = absv(Float(2 * pi64) * exactPhase(point(xCoord, yCoord)));

    // form the rotation quaternion
    return vectorNormalize(quatFromRodrigues(rotationAmount * rotationAxis));
}
