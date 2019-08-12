#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "simpleString/simpleString.h"
#include "cfg/cfgInterface.h"

//================================================================
//
// OutputSimpleString
//
// CfgOutputString interface implementation, saving to SimpleString.
//
//================================================================

template <typename Type>
class OutputSimpleString : public CfgOutputString
{

    SimpleStringEx<Type>& str;

public:

    inline OutputSimpleString(SimpleStringEx<Type>& str)
        : str(str) {}

    bool addBuf(const Type* bufArray, size_t bufSize)
    {
        str += SimpleStringEx<Type>(bufArray, bufSize);
        bool ok = def(str);

        if_not (def(str))
            str.clear();

        return ok;
    }
};

//================================================================
//
// SimpleStringVarEx
//
// SimpleString which maintains 'changed' flag.
//
//================================================================

template <typename Type>
class SimpleStringVarEx
{

public:

    template <typename DefaultValue>
    explicit SimpleStringVarEx(const DefaultValue& defaultValue)
    {
        this->value = defaultValue;
        this->defaultValue = defaultValue;
    }

public:

    template <typename AnyString>
    inline void operator =(const AnyString& X)
    {
        if_not (value == X)
        {
            value = X;

            if_not (def(value))
                value.clear();

            changed = true;
        }
    }

public:

    inline void clear()
    {
        if_not (value.isOk() && value.size() == 0)
        {
            value.clear();
            changed = true;
        }
    }

public:

    void resetValue()
    {
        value = defaultValue;
    }

public:

    operator const SimpleStringEx<Type>& () const
        {return value;}

    const SimpleStringEx<Type>& operator()() const
        {return value;}

    const SimpleStringEx<Type>* operator ->() const
        {return &value;}

public:

    //
    // Returns true if the variable was NOT changed.
    //

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment = STR(""), const CharArray& blockComment = STR(""));

    template <typename T>
    friend class SerializeSimpleStringVar;

private:

    SimpleStringEx<Type> value;
    SimpleStringEx<Type> defaultValue;

    bool changed = false;

};

//================================================================
//
// SerializeSimpleStringVar
//
//================================================================

template <typename Type>
class SerializeSimpleStringVar : public CfgSerializeVariable
{

public:

    inline SerializeSimpleStringVar
    (
        SimpleStringVarEx<Type>& baseVar,
        const CharArray& nameDesc,
        const CharArray& comment,
        const CharArray& blockComment
    )
        :
        baseVar(baseVar),
        nameDesc(nameDesc),
        comment(comment),
        blockComment(blockComment)
    {
    }

private:

    bool changed() const
        {return baseVar.changed;}

    void clearChanged() const
        {baseVar.changed = false;}

    void resetValue() const
        {baseVar.resetValue();}

    bool getName(CfgOutputString& result) const
    {
        ensure(nameDesc.size != 0);
        ensure(result.addStr(nameDesc));
        return true;
    }

    bool getTextValue(CfgWriteStream& s) const
    {
        ensure(s.writeStr(charArrayFromPtr(baseVar->cstr())));
        return true;
    }

    bool setTextValue(CfgReadStream& s) const
    {
        SimpleStringEx<Type> tmp;
        OutputSimpleString<Type> output(tmp);
        ensure(s.readString(output));
        ensure(def(tmp));
        baseVar = tmp;
        return true;
    }

    bool getTextComment(CfgWriteStream& s) const
    {
        ensure(comment.size != 0);
        ensure(cfgWrite(s, comment));
        return true;
    }

    bool getBlockComment(CfgWriteStream& s) const
    {
        ensure(cfgWrite(s, blockComment));
        return true;
    }

private:

    SimpleStringVarEx<Type>& baseVar;

    const CharArray nameDesc;
    const CharArray comment;
    const CharArray blockComment;

};

//================================================================
//
// SimpleStringVarEx<Type>::serialize
//
//================================================================

template <typename Type>
inline bool SimpleStringVarEx<Type>::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment, const CharArray& blockComment)
{
    auto oldValue = value;
    SerializeSimpleStringVar<Type> serializeVar(*this, name, comment, blockComment);
    kit.visitor(kit.scope, serializeVar);
    return allv(oldValue == value);
}

//================================================================
//
// SimpleStringVar
//
//================================================================

using SimpleStringVar = SimpleStringVarEx<CharType>;
