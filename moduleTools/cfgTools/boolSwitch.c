#include "boolSwitch.h"

//================================================================
//
// SerializeBoolSwitchVariable
//
//================================================================

class SerializeBoolSwitchVariable : public SerializeBoolVar
{

    const CharArray comment;
    const CharArray key;

public:

    inline SerializeBoolSwitchVariable(NumericVar<int32>& targetVar, const CharArray& name, const CharArray& comment, const CharArray& key)
        :
        SerializeBoolVar(targetVar, name, STR(""), STR("")),
        comment(comment),
        key(key)
    {
    }

    bool getTextComment(CfgWriteStream& s) const
    {
        if (comment.size != 0)
            cfgWrite(s, comment);

        if (key.size)
        {
            if (comment.size != 0)
                cfgWrite(s, STR(", "));

            cfgWrite(s, STR("["));
            cfgWrite(s, key);
            cfgWrite(s, STR("]"));
        }

        return true;
    }
};

//================================================================
//
// SerializeRingSwitchSignal
//
//================================================================

class SerializeRingSwitchSignal : public SerializeStandardSignal
{

    const CharArray namePostfix;

public:

    inline SerializeRingSwitchSignal(StandardSignal& baseSignal, const CharArray& name, const CharArray& namePostfix, const CharArray& key, const CharArray& comment)
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
// BoolSwitch::serialize
//
//================================================================

bool BoolSwitch::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key, const CharArray& comment)
{
    int32 oldValue = base;

    SerializeRingSwitchSignal serializeSignal(signal, name, STR(" <->"), key, comment);
    kit.visitSignal(serializeSignal);

    SerializeBoolSwitchVariable serializeVar(base, name, comment, key);
    kit.visitVar(serializeVar);

    if (signal)
    {
        base = (base() ^ (signal & 1)) != 0;
        signal.clear();
    }

    return base == oldValue;
}

//================================================================
//
// SerializeRingSwitchVariable
//
//================================================================

class SerializeRingSwitchVariable : public SerializeNumericVar<int32>
{

    const CharArray comment;
    const CharArray key;

public:

    inline SerializeRingSwitchVariable(NumericVar<int32>& targetVar, const CharArray& name, const CharArray& comment, const CharArray& key)
        :
        SerializeNumericVar<int32>(targetVar, name, STR(""), STR("")),
        comment(comment),
        key(key)
    {
    }

    bool getTextComment(CfgWriteStream& s) const
    {
        if (comment.size != 0)
            cfgWrite(s, comment);

        if (key.size)
        {
            if (comment.size != 0)
                cfgWrite(s, STR(", "));

            cfgWrite(s, STR("["));
            cfgWrite(s, key);
            cfgWrite(s, STR("]"));
        }

        return true;
    }
};

//================================================================
//
// RingSwitchBase::RingSwitchBase
//
//================================================================

RingSwitchBase::RingSwitchBase(int32 positionCount, int32 defaultPosition)
{
    positionCount = clampMin(positionCount, 1);
    defaultPosition = clampRange(defaultPosition, 0, positionCount-1);
    value.setup(0, positionCount-1, defaultPosition);
}

//================================================================
//
// RingSwitchBase::serialize
//
//================================================================

bool RingSwitchBase::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& key, const CharArray& comment)
{
    int32 oldValue = value;

    SerializeRingSwitchSignal serializeSignal(signal, name, STR(" Cycle"), key, comment);
    kit.visitSignal(serializeSignal);

    SerializeRingSwitchVariable serializeVar(value, name, comment, key);
    kit.visitVar(serializeVar);

    if (signal)
    {
        value = (value + signal) % (value.maxValue() + 1);
        signal.clear();
    }

    return value == oldValue;
}
