#if HOSTCODE
#include "visualizeVectorImage.h"
#endif

#include "numbers/mathIntrinsics.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"
#include "vectorTypes/vectorOperations.h"
#include "readInterpolate/gpuTexCubic.h"
#include "types/lt/ltType.h"
#include "gpuSupport/gpuTexTools.h"
#include "mathFuncs/rotationMath.h"
#include "mathFuncs/gaussApprox.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "numbers/mathIntrinsics.h"
#include "imageRead/positionTools.h"
#include "computeVectorVisualization.h"

//================================================================
//
// FOREACH_INTERP_BORDER
//
//================================================================

#define FOREACH_INTERP_BORDER(action, extra) \
    INTERP_TYPE_FOREACH(FOREACH_INTERP_BORDER_AUX1, (action, extra))

#define FOREACH_INTERP_BORDER_AUX1(interpType, params) \
    FOREACH_INTERP_BORDER_AUX2(interpType, PREP_ARG2_0 params, PREP_ARG2_1 params)

#define FOREACH_INTERP_BORDER_AUX2(interpType, action, extra) \
    BORDER_MODE_FOREACH(FOREACH_INTERP_BORDER_AUX3, (action, interpType, extra))

#define FOREACH_INTERP_BORDER_AUX3(borderMode, params) \
    FOREACH_INTERP_BORDER_AUX4(PREP_ARG3_0 params, PREP_ARG3_1 params, borderMode, PREP_ARG3_2 params)

#define FOREACH_INTERP_BORDER_AUX4(action, interpType, borderMode, extra) \
    action(interpType, borderMode, extra)

//================================================================
//
// FOREACH_VISUALIZATION_TYPE
//
//================================================================

#define FOREACH_VISUALIZATION_TYPE(action) \
    \
    FOREACH_INTERP_BORDER(action, float16_x2) \
    FOREACH_INTERP_BORDER(action, float16_x4) \
    \
    FOREACH_INTERP_BORDER(action, float32_x2) \
    FOREACH_INTERP_BORDER(action, float32_x4)

//================================================================
//
// visualizeVectorImage16
// visualizeVectorImage32
//
//================================================================

#define TMP_MACRO(interpType, borderMode, VectorType) \
    \
    GPUTOOL_2D \
    ( \
        visualize_##VectorType##_##interpType##_##borderMode, \
        ((const VectorType, src, interpType, borderMode)), \
        ((uint8_x4, dst)), \
        ((LinearTransform<Point<float32>>, coordBackTransform)) \
        ((float32, vectorFactor)) \
        ((bool, grayMode)), \
        \
        { \
            Point<float32> srcPos = coordBackTransform(point(Xs, Ys)); \
            \
            auto value = vectorFactor * devTex2D(srcSampler, srcPos.X * srcTexstep.X, srcPos.Y * srcTexstep.Y); \
            \
            if (interpType == INTERP_CUBIC) \
                value = vectorFactor * tex2DCubic(srcSampler, srcPos, srcTexstep); \
            \
            auto color = computeVectorVisualization(make_float32_x2(value.x, value.y), grayMode); \
            color = limitColorBrightness(color); \
            \
            storeNorm(dst, color); \
        } \
    )

FOREACH_VISUALIZATION_TYPE(TMP_MACRO)

#undef TMP_MACRO

//================================================================
//
// visualizeVectorImage
//
//================================================================

#define CALL_KERNEL(VectorType) \
    FOREACH_INTERP_BORDER(CALL_KERNEL_AUX, VectorType)

#define CALL_KERNEL_AUX(i, b, VectorType) \
    if (interpType == i && borderMode == b) \
        require(visualize_##VectorType##_##i##_##b(src, dst, coordBackTransform, vectorFactor, grayMode, stdPassThru));

//----------------------------------------------------------------

#if HOSTCODE

#define TMP_MACRO(VectorType) \
    \
    template <> \
    stdbool visualizeVectorImage \
    ( \
        const GpuMatrix<const VectorType>& src, \
        const GpuMatrix<uint8_x4>& dst, \
        const LinearTransform<Point<float32>>& coordBackTransform, \
        float32 vectorFactor, \
        InterpType interpType, \
        BorderMode borderMode, \
        bool grayMode, \
        stdPars(GpuProcessKit) \
    ) \
    { \
        CALL_KERNEL(VectorType) \
        returnTrue; \
    }

TMP_MACRO(float16_x2)
TMP_MACRO(float16_x4)

TMP_MACRO(float32_x2)
TMP_MACRO(float32_x4)

#endif

//================================================================
//
// renderSigmoid
//
//================================================================

sysinline float32 renderSigmoid(float32 x)
{
    return sigmoidApprox<4>((1/0.3f) * x);
}

//================================================================
//
// imposeVectorArrow
//
//================================================================

GPUTOOL_2D_BEG
(
    imposeVectorArrow,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((Point<float32>, vectorBegin))
    ((Point<float32>, vectorValue))
    ((bool, orientationMode))
)
#if DEVCODE
{
    VECTOR_DECOMPOSE_EX(vec, vectorValue);

    //
    // For orientation mode, un-double-fold the direction,
    // and optionally rotate 90 degrees to show along contours.
    //

    if (orientationMode)
        vecDir = circleCCW(0.5f * fastPhase(vecDir) + 1 * 0.25f);

    ////

    Point<float32> revPos = complexMul(point(Xs, Ys) - vectorBegin, complexConjugate(vecDir));

    ////

    float32 borderRadius = 1.f;
    float32 divAntialiasSigmaSq = 1 / (0.6f * 0.6f);

    float32 pureDistSq = square(revPos.Y);

    ////

    if_not (orientationMode)
    {
        float32 arrowHeight = 3.0f;

        const float32 arrowLen = vecLength;
        float32 divArrowLen = fastRecipZero(arrowLen);

        ////

        float32 currentArrowHeight = arrowHeight * saturate(divArrowLen * (vecLength - revPos.X));
        float32 maxArrowHeight = arrowHeight;

        ////

        float32 arrowStart = clampMin(vecLength - arrowLen, 0.f);

        ////

        float32 lineHeight = 0.5f;


        float32 lineDistanceSq = vectorLengthSq(revPos - point(clampRange(revPos.X, 0.f, arrowStart), clampRange(revPos.Y, -lineHeight, +lineHeight)));
        float32 arrowDistSq = vectorLengthSq(revPos - point(clampRange(revPos.X, arrowStart, vecLength), clampRange(revPos.Y, -currentArrowHeight, +currentArrowHeight)));
        float32 circleDistSq = square(clampMin(vectorLength(revPos) - maxArrowHeight, 0.f));
        pureDistSq = minv(lineDistanceSq, arrowDistSq, circleDistSq);
    }

    ////

    float32 figurePresence = gaussExpoApprox<3>(pureDistSq * divAntialiasSigmaSq);
    float32 shadowPresence = gaussExpoApprox<3>(square(clampMin(fastSqrt(pureDistSq) - borderRadius, 0.f)) * divAntialiasSigmaSq);

    ////

    float32 figurePart = figurePresence;
    float32 shadowPart = (1 - figurePresence) * shadowPresence;
    float32 imagePart = saturate(1 - figurePart - shadowPart);

    ////

    auto imageValue = loadNorm(dst);
    storeNorm(dst, imagePart * imageValue + shadowPart * zeroOf<float32_x4>() + figurePart * make_float32_x4(1, 1, 1, 0));

}
#endif
GPUTOOL_2D_END
