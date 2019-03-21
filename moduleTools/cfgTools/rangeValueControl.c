#include "rangeValueControl.h"

#include "numbers/float/floatType.h"

//================================================================
//
// RangeValueControl::serialize
//
//================================================================

template <typename Type>
bool RangeValueControl<Type>::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& keyDec, const CharArray& keyInc, const CharArray& keyReset, bool* signalActivity)
{
    Type oldValue = value;

    ////

    value.serialize(kit, name);

    signalDec.serialize(kit, STR(""), keyDec);
    signalInc.serialize(kit, STR(""), keyInc);
    signalReset.serialize(kit, STR(""), keyReset);

    ////

    if (controlType == RangeValueLinear)
    {
        for (int32 i = 0; i < signalInc; ++i)
            value = value + increment;

        for (int32 i = 0; i < signalDec; ++i)
            value = value - increment;
    }

    ////

    if (controlType == RangeValueCircular)
    {
        Type tmpValue = value;

        for (int32 i = 0; i < signalInc; ++i)
            tmpValue += increment;

        for (int32 i = 0; i < signalDec; ++i)
            tmpValue -= increment;

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

            for (int32 i = 0; i < signalInc; ++i)
                value = value * incFactor;

            for (int32 i = 0; i < signalDec; ++i)
                value = value * decFactor;
        }
    }
  
    ////

    if (signalReset)
        value = value.defaultValue();

    ////

    bool activity = signalDec || signalInc || signalReset;
    if (signalActivity) *signalActivity = activity;

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
