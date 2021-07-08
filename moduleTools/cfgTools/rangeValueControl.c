#include "rangeValueControl.h"

#include "numbers/float/floatType.h"

//================================================================
//
// RangeValueControl<Type>::feedIncrements
//
//================================================================

template <typename Type>
void RangeValueControl<Type>::feedIncrements(int32 sigDec, int32 sigInc, int32 sigReset)
{
    if (controlType == RangeValueLinear)
    {
        for_count (i, sigDec)
            value = value - increment;

        for_count (i, sigInc)
            value = value + increment;
    }

    ////

    if (controlType == RangeValueCircular)
    {
        Type tmpValue = value;

        for_count (i, sigDec)
            tmpValue -= increment;

        for_count (i, sigInc)
            tmpValue += increment;

        Type period = value.maxValue() - value.minValue();
    
        while_not (tmpValue < value.maxValue())
            tmpValue -= period;

        while_not (tmpValue >= value.minValue())
            tmpValue += period;

        value = tmpValue;
    }

    ////

    if (controlType == RangeValueLogscale)
    {
        if (increment > 0)
        {
            Type incFactor = increment;
            Type decFactor = 1 / increment;

            for_count (i, sigDec)
                value = value * decFactor;

            for_count (i, sigInc)
                value = value * incFactor;
        }
    }
  
    ////

    if (sigReset)
        value = value.defaultValue();
}

//================================================================
//
// RangeValueControl::serialize
//
//================================================================

template <typename Type>
bool RangeValueControl<Type>::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& keyDec, const CharArray& keyInc, const CharArray& keyReset)
{
    Type oldValue = value;

    ////

    value.serialize(kit, name);

    signalDec.serialize(kit, STR(""), keyDec);
    signalInc.serialize(kit, STR(""), keyInc);
    signalReset.serialize(kit, STR(""), keyReset);

    ////

    feedIncrements(signalDec, signalInc, signalReset);

    ////

    signalDec.clear();
    signalInc.clear();
    signalReset.clear();

    ////

    return (value == oldValue);
}

//================================================================
//
// RangeValueControl instances
//
//================================================================

template class RangeValueControl<int32>;
template class RangeValueControl<uint32>;
template class RangeValueControl<float32>;
