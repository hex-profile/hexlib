#include "gpuSupport/gpuTool.h"
#include "mathFuncs/rotationMath.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"
#include "gpuSupport/gpuTexTools.h"

//================================================================
//
// rotateImage*
//
//================================================================

#define TMP_MACRO(funcName, interpMode, borderMode, sampleTerm) \
    \
    GPUTOOL_2D_AP \
    ( \
        funcName, \
        ((const uint8_x4, src, interpMode, borderMode)), \
        ((uint8_x4, dst)), \
        ((Point<float32>, transMul)) ((Point<float32>, transAdd)), \
        \
        { \
            Point<float32> srcPos = complexMul(point(Xs, Ys), transMul) + transAdd; \
            \
            auto result = sampleTerm; \
            \
            storeNorm(dst, result); \
        } \
    )

TMP_MACRO(rotateImageLinearZero, INTERP_LINEAR, BORDER_ZERO, tex2D(srcSampler, srcPos * srcTexstep))
TMP_MACRO(rotateImageLinearMirror, INTERP_LINEAR, BORDER_MIRROR, tex2D(srcSampler, srcPos * srcTexstep))

TMP_MACRO(rotateImageCubicZero, INTERP_NONE, BORDER_ZERO, tex2DCubic(srcSampler, srcPos, srcTexstep))
TMP_MACRO(rotateImageCubicMirror, INTERP_NONE, BORDER_MIRROR, tex2DCubic(srcSampler, srcPos, srcTexstep))
