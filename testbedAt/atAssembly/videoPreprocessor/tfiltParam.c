#include "tfiltParam.h"

//================================================================
//
// npassDecayAlpha
//
//================================================================

float32 npassDecayAlpha(float32 A, float32 N)
{
    if (A == 1)
        return 1;

    float32 divN = 1 / N;

    float32 binp_2divN = powf(2, divN);
    float32 binp_4divN = binp_2divN * binp_2divN;

    return
    (
        1 -

        expf
        (
            logf(1 - A) *
            sqrtf(15) * binp_2divN /
            (binp_4divN - 1) * sqrtf(1 - 1 / binp_4divN)
        )
    );
}

//================================================================
//
// TfiltParam<Type, n>::addPrepare
//
//================================================================

template <typename Type, Space n>
bool TfiltParam<Type, n>::addPrepare(const TempoScale& periodUp, const TempoScale& periodDn, TfiltSpeed& tfiltSpeed, float32 frameRate)
{
    float32 betaUp = npassDecayAlpha(periodUp.alpha(frameRate), n);
    ensure(def(betaUp));

    float32 betaDn = npassDecayAlpha(periodDn.alpha(frameRate), n);
    ensure(def(betaDn));

    tfiltSpeed.betaUp = betaUp;
    tfiltSpeed.betaDn = betaDn;

    return true;
}

//================================================================
//
// TfiltParam<Type, n>::add
//
//================================================================

template <typename Type, Space n>
bool TfiltParam<Type, n>::add(const Type& value, const TempoScale& periodUp, const TempoScale& periodDn, float32 frameRate)
{
    TfiltSpeed tfiltSpeed;
    ensure(addPrepare(periodUp, periodDn, tfiltSpeed, frameRate));
    addApply(value, tfiltSpeed);

    return true;
}

//================================================================
//
// TfiltParam<Type, n>::assign
//
//================================================================

template <typename Type, Space n>
void TfiltParam<Type, n>::operator =(const TfiltParam<Type, n>& that)
{
    for_count (i, n)
        this->memory[i] = that.memory[i];
}

//================================================================
//
// TfiltNorm<Type, n>::TfiltNorm()
//
//================================================================

template <typename Type, Space n>
TfiltNorm<Type, n>::TfiltNorm()
    :
    avgV(convertNearest<Type>(0)),
    avg1(convertNearest<Single>(0))
{
}

//================================================================
//
// TfiltNorm<Type, n>::reset
//
//================================================================

template <typename Type, Space n>
void TfiltNorm<Type, n>::reset()
{
    avgV.initialize(convertNearest<Type>(0));
    avg1.initialize(convertNearest<Single>(0));
}

//================================================================
//
// TfiltNorm<Type, n>::add
//
//================================================================

template <typename Type, Space n>
bool TfiltNorm<Type, n>::add(const Type& value, const TempoScale& periodUp, const TempoScale& periodDn, float32 frameRate)
{
    ensure(avgV.add(value, periodUp, periodDn, frameRate));
    ensure(avg1.add(convertNearest<Single>(1), periodUp, periodDn, frameRate));

    return true;
}

//================================================================
//
// TfiltNorm<Type, n>::operator () ()
//
//================================================================

template <typename Type, Space n>
Type TfiltNorm<Type, n>::operator () () const
{
    Single denom = avg1();

    return denom > convertNearest<Single>(0) ?
        avgV() / denom :
        convertNearest<Type>(0);
}

//================================================================
//
// TfiltNorm<Type, n>::assign
//
//================================================================

template <typename Type, Space n>
void TfiltNorm<Type, n>::operator =(const TfiltNorm<Type, n>& that)
{
    this->avgV = that.avgV;
    this->avg1 = that.avg1;
}

//================================================================
//
// instance
//
//================================================================

#define TFILT_INST(Type, n) \
    template class TfiltParam<Type, n>; \
    template class TfiltNorm<Type, n>; \

TFILT_INST(float32, 8);
TFILT_INST(float32, 4);
TFILT_INST(float32, 2);
TFILT_INST(float32, 1);

#undef TFILT_INST
