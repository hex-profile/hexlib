#include "temporalParamFilter.h"

#include "numbers/mathIntrinsics.h"
#include "point/point.h"

namespace temporalParamFilter {

//================================================================
//
// halfLifePoints
//
// [exponentialFilterAnalysis.mws]
//
//================================================================

static const float32 halfLifePoints[] =
{
    0.f,
    0.69314718f,
    1.67834699f,
    2.67406031f,
    3.67206075f,
    4.67090888f,
    5.67016119f,
    6.66963708f,
    7.66924944f,
    8.66895118f,
    9.66871462f,
    10.66852240f,
    11.66836315f,
    12.66822906f,
    13.66811460f,
    14.66801576f,
    15.66792954f,
};

//================================================================
//
// getMultiStageAlpha
//
//================================================================

bool getMultiStageAlpha(float32 desiredHalfLife, Space stageCount, float32& result)
{
    ensure(stageCount >= 0 && stageCount < COMPILE_ARRAY_SIZE(halfLifePoints));
    float32 unscaledHalfLifePoint = halfLifePoints[stageCount];

    ////

    float32 divScale = unscaledHalfLifePoint / desiredHalfLife;
    float32 mu = expf(-divScale);

    result = 1 - mu;
    ensure(def(result) && result >= 0.f && result <= 1.f);

    return true;
}

//================================================================
//
// TemporalFactor::TemporalFactor
//
//================================================================

TemporalFactor::TemporalFactor(float32 halfLife, Space stageCount)
{
    if_not (getMultiStageAlpha(halfLife, stageCount, beta))
        beta = float32Nan();
}

//================================================================
//
// TemporalFilter::initialize
//
//================================================================

template <typename Type, int n>
void TemporalFilter<Type, n>::initialize(const Type& value)
{
    for (Space i = 0; i < stageCount; ++i)
        memory[i] = value;
}

//================================================================
//
// TemporalFilter::add
//
//================================================================

template <typename Type, int n>
void TemporalFilter<Type, n>::add(const Type& value, const TemporalFactor& factor)
{
    Type input = value;

    float32 beta = factor.beta;

    for (Space i = 0; i < stageCount; ++i)
    {
        Type& accum = memory[i];
        accum = accum + (input - accum) * beta;
        input = accum;
    }
}

//================================================================
//
// TemporalFilter::addAsymmetric
//
// In case of vector type, the function doesn't make any sense
// and shoudn't be used.
//
//================================================================

template <typename Type, int n>
void TemporalFilter<Type, n>::addAsymmetric(const Type& value, const TemporalFactor& factorUp, const TemporalFactor& factorDn)
{
    Type currentValue = operator () ();
    return add(value, allv(value >= currentValue) ? factorUp : factorDn);
}

//================================================================
//
// TemporalFilter<Type, n>::addContinuous
//
//================================================================

#if 0

template <typename Type, int n>
void TemporalFilter<Type, n>::addContinuous(const Type& value, float32 dTime, const TemporalFactor& factor)
{
    if_not (def(dTime))
        dTime = 0;

    dTime = clampMin(dTime, 0.f);

    ////

    TemporalFactor actualFactor = factor;
    actualFactor.beta = expf(factor.lnBeta * dTime);

    ////

    add(value, actualFactor);
}

#endif

//================================================================
//
// TemporalWeightedFilter<Type, n>::operator ()
//
//================================================================

template <typename Type, int n>
Type TemporalWeightedFilter<Type, n>::operator () () const
{
    return nativeRecip(avgWeight()) * avgWeightValue();
}

//================================================================
//
// instance
//
//================================================================

#define TMP_MACRO_EX(Type, stageCount) \
    template class TemporalFilter<Type, stageCount>; \
    template class TemporalWeightedFilter<Type, stageCount>; \
    template class TemporalFilterNorm<Type, stageCount>; \

#define TMP_MACRO(Type, stageCount) \
    TMP_MACRO_EX(Type, stageCount); \
    TMP_MACRO_EX(Point<Type>, stageCount); \

TMP_MACRO(float32, 8);
TMP_MACRO(float32, 4);
TMP_MACRO(float32, 2);
TMP_MACRO(float32, 1);

#undef TMP_MACRO
#undef TMP_MACRO_EX

//----------------------------------------------------------------

}
