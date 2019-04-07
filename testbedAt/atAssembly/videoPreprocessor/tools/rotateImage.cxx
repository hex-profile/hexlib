#include "gpuSupport/gpuTool.h"
#include "mathFuncs/rotationMath.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"

//================================================================
//
// rotateImage*
//
//================================================================

#define TMP_MACRO(funcName, interpMode, borderMode, sampleTerm) \
    \
    GPUTOOL_2D \
    ( \
        funcName, \
        ((const uint8_x4, src, interpMode, borderMode)), \
        ((uint8_x4, dst)), \
        ((Point<float32>, transMul)) ((Point<float32>, transAdd)), \
        \
        { \
            float32 srcX = complexMulX(Xs, Ys, transMul.X, transMul.Y) + transAdd.X; \
            float32 srcY = complexMulY(Xs, Ys, transMul.X, transMul.Y) + transAdd.Y; \
            \
            float32_x4 result = sampleTerm; \
            \
            storeNorm(dst, result); \
        } \
    )

TMP_MACRO(rotateImageLinearZero, INTERP_LINEAR, BORDER_ZERO, devTex2D(srcSampler, srcX * srcTexstep.X, srcY * srcTexstep.Y))
TMP_MACRO(rotateImageLinearMirror, INTERP_LINEAR, BORDER_MIRROR, devTex2D(srcSampler, srcX * srcTexstep.X, srcY * srcTexstep.Y))

TMP_MACRO(rotateImageCubicZero, INTERP_NONE, BORDER_ZERO, texCubic2D(srcSampler, point(srcX, srcY), srcTexstep))
TMP_MACRO(rotateImageCubicMirror, INTERP_NONE, BORDER_MIRROR, texCubic2D(srcSampler, point(srcX, srcY), srcTexstep))
