#include "mathFuncs/mat4d.h"
#include "types/movement3d/movement3d.h"

//================================================================
//
// mat4dMovement
//
//================================================================

template <typename Float>
sysinline Mat4D<Float> mat4dMovement(const Movement3D<Float>& movement)
{
    auto result = unitMat4D<Float>();

    //----------------------------------------------------------------
    //
    // The 3x3 section in the upper-left of the matrix represents rotation.
    //
    // With a vector (x, y, z) represented as (xw, yw, zw, w), applying
    // the rotation matrix to (xw, yw, zw) and keeping the 4th component w
    // is equivalent to:
    //
    // * Scaling the original 3-D vector (x, y, z) by w,
    // * Applying the rotation matrix R,
    // * Scaling back by (1/w), as the 4th component remains w.
    //
    // Since rotation is scale-independent, the result is a pure rotation.
    //
    //----------------------------------------------------------------

    auto R = quatMat(movement.rotation);

    #define TMP_MACRO(a, b) \
        result.a.b = R.a.b;

    MAT3D_FOREACH(TMP_MACRO)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // The rightmost column of the matrix (excluding the bottom element)
    // is set to the translation vector.
    //
    // As the w component of the original 4D vector is not 1,
    // the xyz components of real vector are scaled by w:
    // (xw, yw, zw, w)
    //
    // The translation vector can be inserted directly,
    // as it will also be scaled by w when applied,
    // thus adding to the xyz components at the same scale.
    //
    //----------------------------------------------------------------

    result.X.W = movement.translation.X;
    result.Y.W = movement.translation.Y;
    result.Z.W = movement.translation.Z;

    ////

    return result;
}
