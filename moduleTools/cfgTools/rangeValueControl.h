#pragma once

#include "cfgTools/standardSignal.h"
#include "cfgTools/numericVar.h"

//================================================================
//
// RangeValueControlType
//
//================================================================

enum RangeValueControlType {RangeValueLinear, RangeValueCircular, RangeValueLogscale};

//================================================================
//
// RangeValueControl
//
//================================================================

template <typename Type>
class RangeValueControl
{

public:

    inline RangeValueControl(const Type& minVal, const Type& maxVal, const Type& defaultVal, const Type& increment, RangeValueControlType controlType)
        : value(minVal, maxVal, defaultVal), increment(increment), controlType(controlType) {}

    //
    // Returns true if the value was NOT changed
    //

    bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& keyDec, const CharArray& keyInc, const CharArray& keyReset = STR(""));

    void feedIncrements(int32 sigDec, int32 sigInc, int32 sigReset);

public:

    inline operator Type() const {return value;}
    inline Type operator()() const {return value;}

public:

    inline RangeValueControl<Type>& operator =(const Type& X)
        {value = X; return *this;}

public:

    #define TMP_MACRO(OP) \
        template <typename AnyType> \
        inline friend bool operator OP(const RangeValueControl<Type>& A, const AnyType& B) \
            {return A.value OP B;}

    TMP_MACRO(==)
    TMP_MACRO(!=)
    TMP_MACRO(>)
    TMP_MACRO(<)
    TMP_MACRO(>=)
    TMP_MACRO(<=)

    #undef TMP_MACRO

private:

    NumericVar<Type> value;

    StandardSignal signalDec;
    StandardSignal signalInc;
    StandardSignal signalReset;

    const Type increment;
    const RangeValueControlType controlType;

};
