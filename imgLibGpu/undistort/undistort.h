#pragma once

#include "gpuProcessHeader.h"

#include "camera/cameraIntrinsics.h"

//================================================================
//
// makeUndistortMap
//
// Takes camera intrinsics in normal (not OpenCV) format!
//
//================================================================

void makeUndistortMap
(
    const CameraIntrinsics<float32>& intrinsics,
    const CameraDistortion<float32>& distortion,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    const GpuMatrix<float32_x2>& map,
    stdPars(GpuProcessKit)
);
