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

    if (absValue * Type(1e-15) >= Type(1))
        {resultValue = value * Type(1e-15); resultUnit = STR("P"); return;}

    if (absValue * Type(1e-12) >= Type(1))
        {resultValue = value * Type(1e-12); resultUnit = STR("T"); return;}

    if (absValue * Type(1e-9) >= Type(1))
        {resultValue = value * Type(1e-9); resultUnit = STR("G"); return;}

    if (absValue * Type(1e-6) >= Type(1))
        {resultValue = value * Type(1e-6); resultUnit = STR("M"); return;}

    if (absValue * Type(1e-3) >= Type(1))
        {resultValue = value * Type(1e-3); resultUnit = STR("K"); return;}
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
