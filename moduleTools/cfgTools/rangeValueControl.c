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
// SerializeSignalWithPostfix
//
//================================================================

class SerializeSignalWithPostfix : public SerializeStandardSignal
{

    const CharArray namePostfix;

public:

    inline SerializeSignalWithPostfix(StandardSignal& baseSignal, const CharArray& name, const CharArray& namePostfix, const CharArray& key, const CharArray& comment)
        :
        SerializeStandardSignal(baseSignal, name, key, comment),
        namePostfix(namePostfix)
    {
    }

    virtual bool getName(CfgOutputString& result) const
    {
        ensure(SerializeStandardSignal::getName(result));

        if (namePostfix.size)
            ensure(result.addStr(namePostfix));

        return true;
    }

};

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


    SerializeSignalWithPostfix serializeDec(signalDec, name, STR(" -"), keyDec, STR(""));
    kit.visitSignal(serializeDec);

    SerializeSignalWithPostfix serializeInc(signalInc, name, STR(" +"), keyInc, STR(""));
    kit.visitSignal(serializeInc);

    SerializeSignalWithPostfix serializeReset(signalReset, name, STR(" Reset"), keyReset, STR(""));
    kit.visitSignal(serializeReset);

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
