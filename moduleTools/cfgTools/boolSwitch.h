#pragma once

#include "cfgTools/standardSignal.h"
#include "cfgTools/numericVar.h"

//================================================================
//
// BoolSwitch
//
//================================================================

class BoolSwitch
{

public:

    inline BoolSwitch(bool value)
        : base(value) {}

public:

    inline operator bool() const {return base != 0;}
    inline bool operator() () const {return base != 0;}

    inline BoolSwitch& operator =(bool value)
        {base = value; return *this;}

    inline void setDefaultValue(bool value)
        {base.setDefaultValue(value);}

    bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key = STR(""), const CharArray& comment = STR(""));

private:

    BoolVar base;
    StandardSignal signal;

};

//================================================================
//
// RingSwitchBase
//
//================================================================

class RingSwitchBase
{

public:

    RingSwitchBase(int32 positionCount, int32 defaultPosition);

public:

    inline int32 getValue() const {return value;}
    inline void setValue(int32 newValue) {value = newValue;}

public:

    bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key = STR(""), const CharArray& comment = STR(""));

private:

    NumericVar<int32> value;
    StandardSignal signal;

};

//================================================================
//
// RingSwitch
//
// A codeless wrapper around RingSwitchBase.
//
//================================================================

template <typename EnumType, EnumType positionCount, EnumType defaultPos = EnumType(0)>
class RingSwitch : public RingSwitchBase
{

    COMPILE_ASSERT(int32(positionCount) >= 1 && 0 <= int32(defaultPos) && int32(defaultPos) <= int32(positionCount) - 1);

public:

    inline RingSwitch()
        : RingSwitchBase(int32(positionCount), int32(defaultPos)) {}

public:

    inline operator EnumType() const {return (EnumType) getValue();}
    inline EnumType operator()() const {return (EnumType) getValue();}
    inline void operator =(EnumType newValue) {setValue(int32(newValue));}

};
