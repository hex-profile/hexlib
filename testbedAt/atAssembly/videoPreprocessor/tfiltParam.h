#pragma once

#include "numbers/float/floatType.h"
#include "data/space.h"

//================================================================
//
// TempoScale
//
//================================================================

class TempoScale
{

public:

    // set the filter scale as a half-life period
    TempoScale(float32 halflife) : halflife(halflife) {}

    inline float32 alpha(float32 rate) const
    {
        // constant is -ln(2)
        return 1 - expf(-0.6931471805599453f / (halflife * rate));
    }

private:

    float32 halflife;

};

//================================================================
//
// TfiltSpeed
//
//================================================================

class TfiltSpeed
{

    float32 betaUp;
    float32 betaDn;

    template <typename Type, Space n>
    friend class TfiltParam;

public:

    TfiltSpeed() : betaUp(1), betaDn(1) {}

};

//================================================================
//
// TfiltParam
//
//================================================================

template <typename Type, Space n = 8>
class TfiltParam
{

    Type memory[n];

public:

    explicit TfiltParam(const Type& value)
    {
        initialize(value);
    }

    void initialize(const Type& value)
    {
        for_count (i, n)
            memory[i] = value;
    }

    bool add(const Type& value, const TempoScale& periodUp, const TempoScale& periodDn, float32 frameRate = 1);

    static bool addPrepare(const TempoScale& periodUp, const TempoScale& periodDn, TfiltSpeed& tfiltSpeed, float32 frameRate = 1);

    void addApply(const Type& value, const TfiltSpeed& tfiltSpeed)
    {
        Type input = value;

        Type currentValue = operator () ();
        float32 beta = (input >= currentValue) ? tfiltSpeed.betaUp : tfiltSpeed.betaDn;

        for_count (i, n)
        {
            Type& accum = memory[i];
            accum = accum + (input - accum) * beta;
            input = accum;
        }
    }

    Type operator () () const
    {
        COMPILE_ASSERT(n >= 1);
        return memory[n - 1];
    }

    void operator =(const TfiltParam<Type, n>& that);

};

//================================================================
//
// TfiltNorm
//
//================================================================

template <typename Type, Space n = 8>
class TfiltNorm
{

    using Single = typename VECTOR_BASE(Type);

    TfiltParam<Type, n> avgV;
    TfiltParam<Single, n> avg1;

public:

    TfiltNorm();
    void reset();

    static sysinline bool addPrepare(const TempoScale& periodUp, const TempoScale& periodDn, TfiltSpeed& tfiltSpeed, float32 frameRate = 1)
        {return TfiltParam<Type, n>::addPrepare(periodUp, periodDn, tfiltSpeed, frameRate = 1);}

    bool add(const Type& value, const TempoScale& periodUp, const TempoScale& periodDn, float32 frameRate = 1);

    void addApply(const Type& value, const TfiltSpeed& tfiltSpeed)
    {
        avgV.addApply(value, tfiltSpeed);
        avg1.addApply(convertNearest<Single>(1), tfiltSpeed);
    }

    Single maturity() const {return avg1();}

    Type operator () () const;

    void operator =(const TfiltNorm<Type, n>& that);

};
