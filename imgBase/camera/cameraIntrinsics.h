#pragma once

#include "extLib/types/floatBase.h"
#include "extLib/types/pointTypes.h"
#include "extLib/types/compileTools.h"

//================================================================
//
// CameraIntrinsics
//
// Application of camera intrinsics is:
// (u, v) = (x/z, y/z) * focal + center
//
// focal = (L / W) * resolution,
// where L is the distance from the eye to the screen and W is the screen size.
//
// L / W = 1 / (2 * tan(fov / 2))
//
// where L is the distance from the eye to the screen and W is the screen size.
//
// focal = (1 / (2 * tan(fov / 2))) * resolution
//
// In normal case, the center is normal coordinates: 0th pixel center is at 0.5.
//
// In OpenCV case, focal is (fx, fy), center is (cx, cy)
// and the center is in "index coordinates": 0th pixel center is at 0.
//
//================================================================

template <typename Float>
struct CameraIntrinsics
{
    HEXLIB_INLINE CameraIntrinsics() {}

    HEXLIB_INLINE explicit CameraIntrinsics(const Point<Float>& focal, const Point<Float>& center)
        : focal(focal), center(center) {}

    Point<Float> focal{0, 0};
    Point<Float> center{0, 0};
};

//----------------------------------------------------------------

template <typename Float>
HEXLIB_INLINE bool completelyEqual(const CameraIntrinsics<Float>& a, const CameraIntrinsics<Float>& b)
{
    return
        a.center.X == b.center.X &&
        a.center.Y == b.center.Y &&
        a.focal.X == b.focal.X &&
        a.focal.Y == b.focal.Y;
}

//================================================================
//
// CameraDistortion
//
// Camera distortion parameters in OpenCV format
// (see initUndistortRectifyMap for reference).
//
//================================================================

static constexpr int cameraDistortionCount = 12;

//----------------------------------------------------------------

template <typename Float>
struct CameraDistortion
{
    Float coeffs[cameraDistortionCount];

    HEXLIB_INLINE CameraDistortion()
    {
        for (int i = 0; i < cameraDistortionCount; ++i)
            coeffs[i] = 0;
    }
};

//----------------------------------------------------------------

template <typename Float>
HEXLIB_INLINE bool completelyEqual(const CameraDistortion<Float>& a, const CameraDistortion<Float>& b)
{
    for (int i = 0; i < cameraDistortionCount; ++i)
        if (a.coeffs[i] != b.coeffs[i])
            return false;

    return true;
}
