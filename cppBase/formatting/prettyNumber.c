#include "prettyNumber.h"

#include "numbers/interface/numberInterface.h"
#include "numbers/float/floatType.h"

//================================================================
//
// reduceValue
//
//================================================================

template <typename Type>
sysinline void reduceValue(const Type& value, Type& resultValue, CharArray& resultUnit)
{
    resultValue = value;
    resultUnit = STR("");

    ////

    Type absValue = absv(value);
    ensurev(def(absValue));

    ////

    if (absValue == 0)
        return;

    ////

    auto checkBig = [&] (auto testFactor, auto testUnit) -> bool
    {
        auto testValue = resultValue * testFactor;

        ensure(absv(testValue) >= Type(1));

        resultValue = testValue;
        resultUnit = testUnit;
        return true;
    };

    ////

    ensurev(!checkBig(Type(1e-15), STR("P")));
    ensurev(!checkBig(Type(1e-12), STR("T")));
    ensurev(!checkBig(Type(1e-9), STR("G")));
    ensurev(!checkBig(Type(1e-6), STR("M")));
    ensurev(!checkBig(Type(1e-3), STR("K")));

    ////

    auto checkSmall = [&] (auto testFactor, auto testUnit) -> bool
    {
        auto testValue = resultValue * testFactor;

        ensure(absv(testValue) < Type(1000));

        resultValue = testValue;
        resultUnit = testUnit;
        return true;
    };

    ////

    ensurev(!checkSmall(Type(1e15), STR("e-15")));
    ensurev(!checkSmall(Type(1e12), STR("e-12")));
    ensurev(!checkSmall(Type(1e9), STR("e-9")));
    ensurev(!checkSmall(Type(1e6), STR("e-6")));
}

//================================================================
//
// doFormatOutput
//
//================================================================

template <typename Type>
static sysinline void doFormatOutput(const PrettyNumber<Type>& value, FormatOutputStream& outputStream)
{
    int32 order = 0;

    Type usedValue;
    CharArray usedUnit;
    reduceValue(value.number.value, usedValue, usedUnit);

    outputStream << formatNumber(usedValue, value.number) << usedUnit;
}

//================================================================
//
// formatOutput<PrettyNumber>
//
//================================================================

template <>
void formatOutput(const PrettyNumber<float32>& value, FormatOutputStream& outputStream)
    {doFormatOutput(value, outputStream);}

template <>
void formatOutput(const PrettyNumber<float64>& value, FormatOutputStream& outputStream)
    {doFormatOutput(value, outputStream);}
