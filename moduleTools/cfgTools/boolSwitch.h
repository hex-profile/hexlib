#ifndef _279352ED9ABAB771
#define _279352ED9ABAB771

#include "cfgTools/standardSignal.h"
#include "cfgTools/numericVar.h"

//================================================================
//
// BoolSwitch
//
//================================================================

template <bool defaultBool>
class BoolSwitch
{

public:

    inline operator bool() const {return value != 0;}
    inline bool operator() () const {return value != 0;}

    inline BoolSwitch<defaultBool>& operator =(bool value)
        {this->value = value; return *this;}

    bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key = STR(""), const CharArray& comment = STR(""));

private:

    BoolVarStatic<defaultBool> value;
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

    COMPILE_ASSERT(positionCount >= 1 && 0 <= defaultPos && defaultPos <= positionCount - 1);

public:

    inline RingSwitch()
        : RingSwitchBase(positionCount, defaultPos) {}

public:

    inline operator EnumType() const {return (EnumType) getValue();}
    inline EnumType operator()() const {return (EnumType) getValue();}
    inline void operator =(EnumType newValue) {setValue(newValue);}

};

//----------------------------------------------------------------

#endif // _279352ED9ABAB771
