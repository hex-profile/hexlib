#include "multiSwitch.h"

//================================================================
//
// SerializeMultiSwitchIntVariable
//
//================================================================

class SerializeMultiSwitchIntVariable : public SerializeNumericVar<size_t>
{

    size_t const positionCount;
    const NameKeyCommentStruct* const descPtr;
    bool const prefix;

public:

    inline SerializeMultiSwitchIntVariable(NumericVar<size_t>& targetVar, const CharArray& nameDesc, size_t positionCount, const NameKeyCommentStruct descPtr[], bool prefix)
        : SerializeNumericVar(targetVar, nameDesc, STR(""), STR("")), descPtr(descPtr), prefix(prefix), positionCount(positionCount) {}

    size_t estimateApproxLength() const
    {
        size_t length = 0;

        for (size_t k = 0; k < positionCount; ++k)
        {
            if (k != 0)
                length += 2; // separator ", "

            if (prefix)
                length += 3; // assume 2 digits and space

            length += descPtr[k].name.size;

            const CharArray& key = descPtr[k].key;

            if (key.size)
            {
                length += key.size;
                length += 9; // "Press" etc
            }
        }

        return length;
    }

    bool getComment(bool block, CfgWriteStream& s) const
    {
        for (size_t k = 0; k < positionCount; ++k)
        {
            if (k != 0)
                require(cfgWrite(s, !block ? STR(", ") : STR("\n")));

            if (prefix)
            {
                require(cfgWrite(s, k));
                require(cfgWrite(s, STR(" ")));
            }

            require(cfgWrite(s, descPtr[k].name));

            const CharArray& key = descPtr[k].key;

            if (key.size)
            {
                cfgWrite(s, STR(" ["));
                cfgWrite(s, key);
                cfgWrite(s, STR("]"));
            }
        }

        return true;
    }

    bool longComment() const {return estimateApproxLength() >= 100;}

    bool getTextComment(CfgWriteStream& s) const
    {
        if_not (longComment())
            require(getComment(false, s));

        return true;
    }

    bool getBlockComment(CfgWriteStream& s) const
    {
        if (longComment())
            require(getComment(true, s));

        return true;
    }

};

//================================================================
//
// SerializeSignalWithPrefix
//
//================================================================

class SerializeSignalWithPrefix : public SerializeStandardSignal
{

    const CharArray prefix;

public:

    inline SerializeSignalWithPrefix
    (
        StandardSignal& baseSignal,
        const CharArray& prefix,
        const CharArray& name,
        const CharArray& key,
        const CharArray& comment
    )
        :
        SerializeStandardSignal(baseSignal, name, key, comment),
        prefix(prefix)
    {
    }

    virtual bool getName(CfgOutputString& result) const
    {
        require(result.addStr(prefix));
        require(result.addStr(STR("/-> ")));
        return SerializeStandardSignal::getName(result);
    }

};

//================================================================
//
// serializeMultiSwitch
//
//================================================================

bool serializeMultiSwitch
(
    const CfgSerializeKit& kit,
    const CharArray& name,
    NumericVar<size_t>& value,
    size_t positionCount,
    StandardSignal signals[],
    const NameKeyCommentStruct descPtr[],
    bool cfgValuePrefix,
    bool signalPrefix
)
{
    size_t oldValue = value;


    SerializeMultiSwitchIntVariable serializeVar(value, name, positionCount, descPtr, cfgValuePrefix);
    kit.visitor(kit.scope, serializeVar);

    for (size_t k = 0; k < positionCount; ++k)
    {
        if (signalPrefix)
        {
            SerializeSignalWithPrefix serializeSignal(signals[k], name, descPtr[k].name, descPtr[k].key, descPtr[k].comment);
            kit.visitor(kit.scope, serializeSignal);
        }
        else
        {
            signals[k].serialize(kit, descPtr[k].name, descPtr[k].key, descPtr[k].comment);
        }
    }

    for (size_t k = 0; k < positionCount; ++k)
    {
        if (signals[k])
            value = k;

        signals[k].clear();
    }

    return (value == oldValue);
}
