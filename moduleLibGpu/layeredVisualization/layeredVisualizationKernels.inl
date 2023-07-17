#define LAYERS PREP_ITER_INDEX_0

//================================================================
//
// renderVectorSet
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE_UNDER4(renderVectorSet, LAYERS, VectorType, PresenceType),
    GPUTOOL_INDEXED_SAMPLER(LAYERS, const VectorType, srcVector, INTERP_NONE, BORDER_CLAMP)
    GPUTOOL_INDEXED_SAMPLER(LAYERS, const PresenceType, srcPresence, INTERP_NONE, BORDER_CLAMP),
    ((uint8_x4, dst)),
    ((bool, independentPresenceMode))
    ((Point<float32>, srcPos))
    ((float32, vectorFactor))
    ((float32, thicknessRadius))
)
#if DEVCODE
{

    Point<float32> rectRadius = 0.5f * convertFloat32(vGlobSize);
    Point<float32> divBorderedRectRadius = 1.f / (rectRadius - thicknessRadius);
    float32 divThicknessRadius2 = square(fastRecipZero(thicknessRadius));

    Point<float32> pos = point(Xs, Ys) - rectRadius;

    ////

    float32 minRadius = minv(rectRadius.X, rectRadius.Y);

    ////

    auto sumWeightColor = zeroOf<float32_x4>();
    float32 sumWeight = 0;
    float32 maxWeight = 0;

    #define TMP_MACRO(k, o) \
        { \
            auto vecPure = vectorFactor * tex2D(srcVector##k##Sampler, srcPos * srcVector##k##Texstep); \
            auto vecf = minRadius * vecPure; \
            float32 presence = clampMin(tex2D(srcPresence##k##Sampler, srcPos * srcPresence##k##Texstep), 0.25f); \
            Point<float32> vec = point(vecf.x, vecf.y); \
            float32 factorX = 1.f / clampMin(absv(vec.X) * divBorderedRectRadius.X, 1.f); \
            float32 factorY = 1.f / clampMin(absv(vec.Y) * divBorderedRectRadius.Y, 1.f); \
            vec *= minv(factorX, factorY); \
            \
            float32 spatialWeight = gaussExpoApprox<4>(vectorLengthSq(pos - vec) * divThicknessRadius2); \
            \
            auto pureColor = limitColorBrightness(computeVectorVisualization(8.f * vecPure)); \
            \
            sumWeightColor += spatialWeight * presence * pureColor; \
            sumWeight += spatialWeight * presence; \
            maxWeight = maxv(maxWeight, spatialWeight * presence); \
        }

    PREP_FOR(LAYERS, TMP_MACRO, o)

    #undef TMP_MACRO

    ////

    float32 vectorPresence = saturatev(independentPresenceMode ? maxWeight : sumWeight);
    auto vectorColor = fastRecipZero(sumWeight) * sumWeightColor;

    ////

    float32 axisPresence = gaussExpoApprox<4>(minv(square(2.f * pos.X), square(2.f * pos.Y)));
    auto backgroundColor = linerp(axisPresence, zeroOf<float32_x4>(), make_float32_x4(0.15f, 0.15f, 0.15f, 0));

    ////

    auto result = linerp(vectorPresence, backgroundColor, vectorColor);

    ////

    storeNorm(dst, result);

}
#endif
GPUTOOL_2D_END

//================================================================
//
// upsampleVectorVisualizationFunc
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE_UNDER4(upsampleVectorVisualizationFunc, LAYERS, VectorType, PresenceType),
    GPUTOOL_INDEXED_SAMPLER(LAYERS, const VectorType, srcVector, INTERP_NEAREST, BORDER_MIRROR)
    GPUTOOL_INDEXED_SAMPLER(LAYERS, const PresenceType, srcPresence, INTERP_NEAREST, BORDER_MIRROR),
    ((uint8_x4, dstColor)),
    ((Point<float32>, dstToSrcFactor))
    ((float32, divMaxVector))
    ((bool, upsampleInterpolation))
)
#if DEVCODE
{

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    const float32 upsampleSigma = 0.6f;
    const float32 divUpsampleSigmaSq = 1.f / (upsampleSigma * upsampleSigma);

    // const float32 filterCoverageRadius = 2.f;
    const Space taps = 4; // ceil(2*filterCoverageRadius)

    ////

    Point<float32> srcTexstep = srcVector0Texstep;

    Point<float32> dstPos = point(Xs, Ys);
    Point<float32> srcPos = dstPos * dstToSrcFactor;

    //
    //
    //

    float32 sumWeight = 0;
    float32 sumWeightPresence = 0;
    auto sumWeightColor = zeroOf<float32_x4>();

    ////

    #define LOCAL_ACCUMULATION(r, _) \
        { \
            auto vec = tex2D(srcVector##r##Sampler, readTexPos); \
            float32 localPresence##r = tex2D(srcPresence##r##Sampler, readTexPos); \
            \
            vec *= divMaxVector; \
            \
            auto color = computeVectorVisualization(vec); \
            float32 presence = clampMin(localPresence##r, PRESENCE_EPSILON); \
            localPresenceSum += presence; \
            localPresenceColorSum += presence * color; \
        }

    ////

    if_not (upsampleInterpolation)
    {
        auto srcNearestPos = roundPosToNearestSample(srcPos);
        auto readTexPos = srcNearestPos * srcTexstep;

        ////

        float32 localPresenceSum = 0;
        auto localPresenceColorSum = zeroOf<float32_x4>();

        PREP_FOR(LAYERS, LOCAL_ACCUMULATION, _)

        ////

        sumWeight = 1;
        sumWeightPresence = localPresenceSum;
        sumWeightColor = localPresenceColorSum;
    }
    else
    {
        // Round up to the nearest grid point
        Point<float32> srcFilterStartPos = computeFilterStartPos(srcPos, taps);

        devUnrollLoop
        for_count (iY, taps)
        {
            float32 readY = srcFilterStartPos.Y + iY;
            float32 dY = readY - srcPos.Y;

            devUnrollLoop
            for_count (iX, taps)
            {
                float32 readX = srcFilterStartPos.X + iX;
                float32 dX = readX - srcPos.X;

                float32 distSq = square(dX) + square(dY);
                float32 spatialWeight = gaussExpoApprox<4>(distSq * divUpsampleSigmaSq);

                ////

                Point<float32> readTexPos = point(readX, readY) * srcTexstep;

                ////

                float32 localPresenceSum = 0;
                auto localPresenceColorSum = zeroOf<float32_x4>();

                PREP_FOR(LAYERS, LOCAL_ACCUMULATION, _)

                ////

                sumWeight += spatialWeight;
                sumWeightPresence += spatialWeight * localPresenceSum;
                sumWeightColor += spatialWeight * localPresenceColorSum;
            }
        }
    }

    ////

    float32 divSumWeight = fastRecipZero(sumWeight);

    float32 filteredPresence = divSumWeight * sumWeightPresence;
    auto filteredColor = fastRecipZero(sumWeightPresence) * sumWeightColor;

    ////

    auto resultColor = linerp(saturatev(filteredPresence), make_float32_x4(0.5f, 0.5f, 0.5f, 0), filteredColor);

    ////

    storeNorm(dstColor, limitColorBrightness(resultColor));

    ////

    #undef LOCAL_ACCUMULATION

}
#endif
GPUTOOL_2D_END

//================================================================
//
// convertIndependentPresenceToAdditive
//
//================================================================

GPUTOOL_2D_BEG
(
    PREP_PASTE3(convertIndependentPresenceToAdditive, LAYERS, PresenceType),
    PREP_EMPTY,
    GPUTOOL_INDEXED_NAME(LAYERS, const VectorType, srcVector)
    GPUTOOL_INDEXED_NAME(LAYERS, const PresenceType, srcPresence)
    GPUTOOL_INDEXED_NAME(LAYERS, PresenceType, dstPresence),
    ((float32, rsqVectorProximity))
)
#if DEVCODE
{

    //----------------------------------------------------------------
    //
    // Target independent presence
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(i, _) \
        float32 t##i = saturatev(loadNorm(srcPresence##i)); \
        float32 currentPresence##i = t##i; \
        float32_x2 vector##i = loadNorm(srcVector##i);

    PREP_FOR(LAYERS, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Similarities
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(i, j, _) \
        float32 p##i##j = saturatev(gaussExpoApprox<4>(rsqVectorProximity * vectorLengthSq(vector##i - vector##j)));

    PREP_FOR_2D(LAYERS, LAYERS, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Lateral inhibition
    //
    //----------------------------------------------------------------

#if LAYERS == 1

//#elif LAYERS == 2
//
//  if (p01 >= 0.99f)
//  {
//    float32 avgTarget = 0.5f * (t0 + t1);
//    currentPresence0 = currentPresence1 = 0.5f * avgTarget;
//  }
//  else
//  {
//    float32 divider = fastRecipZero(square(p01) - 1);
//    currentPresence0 = (p01*t1-t0) * divider;
//    currentPresence1 = (p01*t0-t1) * divider;
//
//    currentPresence0 = saturatev(currentPresence0);
//    currentPresence1 = saturatev(currentPresence1);
//  }

#else

    for_count (k, 32)
    {
        #define TMP_MACRO_INNER(j, i) \
            currentResult##i += p##i##j * currentPresence##j;

        #define TMP_MACRO(i, _) \
            float32 currentResult##i = 0; \
            PREP_FOR0(LAYERS, TMP_MACRO_INNER, i)

        PREP_FOR1(LAYERS, TMP_MACRO, _)

        #undef TMP_MACRO
        #undef TMP_MACRO_INNER

        ////

        #define TMP_MACRO(i, _) \
            float32 correctionFactor##i = fastDivide(t##i, currentResult##i); \
            if_not (def(correctionFactor##i)) correctionFactor##i = 1; \
            correctionFactor##i = clampRange(correctionFactor##i, 1/4.f, 4.f); \
            currentPresence##i = currentPresence##i * correctionFactor##i;

        PREP_FOR(LAYERS, TMP_MACRO, _)

        #undef TMP_MACRO
    }

#endif

    //----------------------------------------------------------------
    //
    // Store
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(i, _) \
        storeNorm(dstPresence##i, currentPresence##i);

    PREP_FOR(LAYERS, TMP_MACRO, _)

    #undef TMP_MACRO

}
#endif
GPUTOOL_2D_END

