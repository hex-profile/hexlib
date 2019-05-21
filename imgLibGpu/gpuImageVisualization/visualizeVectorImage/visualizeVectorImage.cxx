#if HOSTCODE
#include "visualizeVectorImage.h"
#endif

#include "numbers/mathIntrinsics.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"
#include "vectorTypes/vectorOperations.h"
#include "readInterpolate/gpuTexCubic.h"
#include "numbers/lt/ltType.h"
#include "gpuSupport/gpuTexTools.h"
#include "mathFuncs/rotationMath.h"
#include "mathFuncs/gaussApprox.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// circularDistance
//
//================================================================

sysinline float32 circularDistance(float32 A, float32 B) // A, B in [0..1) range 
{
    float32 distance = A - B + 1; // [0, 2)
  
    if (distance >= 1) 
        distance -= 1; // [0, 1)

    if (distance >= 0.5f) 
        distance = 1 - distance; // [0, 1/2)

    return distance;
}

//================================================================
//
// colorTent
//
//================================================================

sysinline float32 colorTent(float32 value)
{
    return clampMin(1 - 4*absv(value), 0.f);
}

//================================================================
//
// colorBspline
//
//================================================================

sysinline float32 colorBspline(float32 t)
{
    t = absv(t);

    float32 t2 = square(t);

    float32 result = -21.33333333f * t2 + 1;

    if (t >= 0.125f)
        result = 10.66666667f * t2 - 8*t + 1.5f;

    if (t >= 0.375f)
        result = 0;

    return result;
}

//================================================================
//
// computeVectorVisualization
//
//================================================================

sysinline uint8_x4 computeVectorVisualization(const float32_x2& value, bool grayMode)
{
    //
    // vector to H(S)V
    //

    float32 H = 0;

#if 0

    H = atan2f(value.y, value.x) * (1.f / 2 / pi32);
    if (H < 0) H += 1;

#else

    {
        float32 aX = absv(value.x);
        float32 aY = absv(value.y);

        float32 minXY = minv(aX, aY);
        float32 maxXY = maxv(aX, aY);

        float32 D = nativeDivide(minXY, maxXY);
        if (maxXY == 0) D = 0; /* range [0..1] */

        // Cubic polynom approximation, at interval ends x=0 and x=1 both value and 1st derivative are equal to real function.
        float32 result = (0.1591549430918954f + ((-0.02288735772973838f) + (-0.01126758536215698f) * D) * D) * D;

        if (aY >= aX)
            result = 0.25f - result;

        if (value.x < 0)
            result = 0.5f - result;

        if (value.y < 0)
            result = 1 - result;

        H = result; // range [0..1]
    }

#endif

    ////

    float32 length2 = square(value.x) + square(value.y);
    float32 length = fastSqrt(length2);

    //
    // Color conversion
    //

    #define HV_GET_CHANNEL(C, center) \
        \
        float32 C = colorBspline(circularDistance(H, center)); /* [0, 1] */ \

    HV_GET_CHANNEL(weightR, 0.f/4)
    HV_GET_CHANNEL(weightB, 1.f/4)
    HV_GET_CHANNEL(weightG, 2.f/4)
    HV_GET_CHANNEL(weightY, 3.f/4)

    ////

    float32_x4 colorR = make_float32_x4(0, 0, 1, 0);
    float32_x4 colorB = make_float32_x4(1, 0, 0, 0);
    float32_x4 colorG = make_float32_x4(0, 1, 0, 0);
    float32_x4 colorY = make_float32_x4(0, 0.5f, 0.5f, 0);

    float32_x4 color = 
        weightR * colorR + 
        weightB * colorB +
        weightG * colorG +
        weightY * colorY;

    if (grayMode)
        color = make_float32_x4(1, 1, 1, 0);

    ////

    color *= length;

    ////

    float32 maxComponent = maxv(color.x, color.y, color.z);

    if (maxComponent > 1.f)
    {
        float32 divMaxComponent = 1 / maxComponent;
        color *= divMaxComponent;
    }

    ////

    if_not (def(value.x) && def(value.y))
        color = make_float32_x4(1, 0, 1, 0); // magenta

    ////

    return convertNormClamp<uint8_x4>(color);
}

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
            Point<float32> srcPos = ltApply(point(Xs, Ys), coordBackTransform); \
            auto value = vectorFactor * devTex2D(srcSampler, srcPos.X * srcTexstep.X, srcPos.Y * srcTexstep.Y); \
            if (interpType == INTERP_CUBIC) value = vectorFactor * texCubic2D(srcSampler, srcPos, srcTexstep); \
            *dst = computeVectorVisualization(make_float32_x2(value.x, value.y), grayMode); \
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
)
#if DEVCODE
{
    VECTOR_DECOMPOSE_EX(vec, vectorValue);
    Point<float32> revPos = complexMul(point(Xs, Ys) - vectorBegin, conjugate(vecDir));

    ////

    float32 borderRadius = 1.f;
    float32 divAntialiasSigmaSq = 1 / (0.6f * 0.6f);

    ////

    const bool alternativeImpl = true;

    float32 arrowHeight = 3.0f;

    const float32 arrowLen = alternativeImpl ? vecLength : 12.f;
    float32 divArrowLen = nativeRecipZero(arrowLen);

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

    float32 pureDistSq = minv(lineDistanceSq, arrowDistSq, circleDistSq);

    ////

    float32 figurePresence = gaussExpoApprox<3>(pureDistSq * divAntialiasSigmaSq);
    float32 shadowPresence = gaussExpoApprox<3>(square(clampMin(fastSqrt(pureDistSq) - borderRadius, 0.f)) * divAntialiasSigmaSq);

    ////

    float32 figurePart = figurePresence;
    float32 shadowPart = (1 - figurePresence) * shadowPresence;
    float32 imagePart = saturate(1 - figurePart - shadowPart);

    ////

    float32_x4 imageValue = loadNorm(dst);
    storeNorm(dst, imagePart * imageValue + shadowPart * make_float32_x4(0, 0, 0, 0) + figurePart * make_float32_x4(1, 1, 1, 0));

}
#endif
GPUTOOL_2D_END
