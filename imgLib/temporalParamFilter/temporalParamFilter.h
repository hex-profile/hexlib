#pragma once

#include "numbers/float/floatType.h"
#include "data/space.h"

namespace temporalParamFilter {

//================================================================
//
// defaultStages
//
//================================================================

static const Space defaultStages = 8;

//================================================================
//
// TemporalFactor
//
// Temporal factor prepared for fast computations.
//
//================================================================

class TemporalFactor
{

public:

    TemporalFactor() =default;

    TemporalFactor(float32 halfLife, Space stageCount);

private:

    template <typename Type, Space stageCount>
    friend class TemporalFilter;

private:

    float32 beta = 1;

};

//================================================================
//
// TemporalFilter
//
//================================================================

template <typename Type, int n>
class TemporalFilter
{

public:

    static constexpr Space stageCount = n;

    inline TemporalFilter()
    {
        initialize(convertNearest<Type>(0));
    }

    inline explicit TemporalFilter(const Type& value)
    {
        initialize(value);
    }

    inline void reset()
    {
        initialize(convertNearest<Type>(0));
    }

    TemporalFilter& operator =(const TemporalFilter& that) =default;

    void initialize(const Type& value);

    void add(const Type& value, const TemporalFactor& factor);

    void addAsymmetric(const Type& value, const TemporalFactor& factorUp, const TemporalFactor& factorDn);

    inline Type operator () () const
    {
        COMPILE_ASSERT(stageCount >= 1);
        return memory[stageCount - 1];
    }

private:

    Type memory[stageCount];

};

//----------------------------------------------------------------

template <typename Type>
using TemporalFilterStd = TemporalFilter<Type, temporalParamFilter::defaultStages>;

//================================================================
//
// TemporalWeightedFilter
//
// Normalizes the filtered value dividing by
// filtered weight value.
//
//================================================================

template <typename Type, int n>
class TemporalWeightedFilter
{

public:

    static constexpr Space stageCount = n;

    using Scalar = VECTOR_BASE(Type);

    TemporalWeightedFilter()
    {
        reset();
    }

    TemporalWeightedFilter& operator =(const TemporalWeightedFilter& that) =default;

    void reset()
    {
        avgWeightValue.initialize(convertNearest<Type>(0));
        avgWeight.initialize(convertNearest<Scalar>(0));
    }

    void add(const Scalar& weight, const Type& value, const TemporalFactor& factor)
    {
        avgWeightValue.add(weight * value, factor);
        avgWeight.add(weight, factor);
    }

    void addProduct(const Scalar& weight, const Type& weightValue, const TemporalFactor& factor)
    {
        avgWeightValue.add(weightValue, factor);
        avgWeight.add(weight, factor);
    }

    Scalar filteredWeight() const
    {
        return avgWeight();
    }

    Type operator () () const;

private:

    TemporalFilter<Type, n> avgWeightValue;
    TemporalFilter<Scalar, n> avgWeight;

};

//================================================================
//
// TemporalFilterNorm
//
// Normalizes the filtered value dividing by one filtered
// with the same temporal filter.
//
//================================================================

template <typename Type, int n>
class TemporalFilterNorm
{

public:

    static constexpr Space stageCount = n;

    using Scalar = VECTOR_BASE(Type);

    TemporalFilterNorm& operator =(const TemporalFilterNorm& that) =default;

    void reset()
        {base.reset();}

    void add(const Type& value, const TemporalFactor& factor)
        {base.add(1.f, value, factor);}

    Scalar maturity() const
        {return base.filteredWeight();}

    Type operator () () const
        {return base();}

private:

    TemporalWeightedFilter<Type, n> base;

};

//----------------------------------------------------------------

}

namespace tpf = temporalParamFilter;
