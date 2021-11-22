#include "undistort.h"

#include "gpuSupport/gpuTool.h"
#include "mathFuncs/rotationMath.h"
#include "camera/cameraIntrinsicsOps.h"

//================================================================
//
// makeUndistortMapFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    makeUndistortMapFunc,
    PREP_EMPTY,
    ((float32_x2, map)),
    ((Point<float32>, mapScaleFactor))
    ((Point<float32>, mapValueFactor))
    ((CameraIntrinsics<float32>, cameraIntrinsics))
    ((CameraIntrinsics<float32>, cameraIntrinsicsInverse))
    ((CameraDistortion<float32>, cameraDistortion))
)
#if DEVCODE
{
    const float32* c = cameraDistortion.coeffs;
    float32 k1 = c[0x0], k2 = c[0x1], p1 = c[0x2], p2 = c[0x3]; 
    float32 k3 = c[0x4], k4 = c[0x5], k5 = c[0x6], k6 = c[0x7];
    float32 s1 = c[0x8], s2 = c[0x9], s3 = c[0xA], s4 = c[0xB];

    ////

    Point<float32> srcPos = mapScaleFactor * point(Xs, Ys);

    Point<float32> pos = cameraIntrinsicsInverse % srcPos;

    ////

    float32 r2 = vectorLengthSq(pos);
    float32 r4 = square(r2);
    float32 r6 = r4 * r2;

    ////

    float32 factorNum = 1 + k1*r2 + k2*r4 + k3*r6;
    float32 factorDen = 1 + k4*r2 + k5*r4 + k6*r6;
    float32 factor = factorNum * nativeRecip(factorDen);

    ////

    float32 x = pos.X;
    float32 y = pos.Y;
    float32 xy2 = 2*x*y;

    float32 xn = x*factor + p1*xy2 + p2*(r2 + 2*x*x) + s1*r2 + s2*r4;
    float32 yn = y*factor + p2*xy2 + p1*(r2 + 2*y*y) + s3*r2 + s4*r4;

    Point<float32> dstPos = cameraIntrinsics % point(xn, yn);

    Point<float32> result = mapValueFactor * (dstPos - srcPos);

    ////

    *map = make_float32_x2(result.X, result.Y);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// makeUndistortMap
//
//================================================================

#if HOSTCODE

stdbool makeUndistortMap
(
    const CameraIntrinsics<float32>& intrinsics,
    const CameraDistortion<float32>& distortion,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    const GpuMatrix<float32_x2>& map,
    stdPars(GpuProcessKit)
)
{
    require
    (
        makeUndistortMapFunc
        (
            map,
            mapScaleFactor,
            mapValueFactor,
            intrinsics,
            ~intrinsics,
            distortion,
            stdPass
        )
    );

    returnTrue;
}

#endif
