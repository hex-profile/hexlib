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

class OutputSimpleString : public CfgOutputString
{

    SimpleString& str;

public:

    inline OutputSimpleString(SimpleString& str)
        : str(str) {}

    bool addBuf(const CharType* bufArray, size_t bufSize)
    {
        str += SimpleString(bufArray, bufSize);
        bool ok = def(str);

        if_not (def(str))
            str.clear();

        return ok;
    }
};

//================================================================
//
// SimpleStringVar
//
// SimpleString which maintains 'changed' flag.
//
//================================================================

class SimpleStringVar
{

public:

    template <typename DefaultValue>
    SimpleStringVar(const DefaultValue& defaultValue)
    {
        this->value = defaultValue;
        this->defaultValue = defaultValue;
    }

public:

    inline void operator =(const SimpleString& X)
    {
        if (value != X)
        {
            value = X;

            if_not (def(value))
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

    operator const SimpleString& () const
        {return value;}

    const SimpleString& operator()() const
        {return value;}

    operator SimpleString& ()
        {return value;}

    SimpleString& operator()()
        {return value;}

    SimpleString* operator ->()
        {return &value;}

    const SimpleString* operator ->() const
        {return &value;}
public:

    //
    // Returns true if the variable was NOT changed.
    //

    inline bool serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment = STR(""), const CharArray& blockComment = STR(""));

    friend class SerializeSimpleStringVar;

private:

    SimpleString value;
    SimpleString defaultValue;

    bool changed = false;

};

//================================================================
//
// CfgSimpleString
//
//================================================================

class SerializeSimpleStringVar : public CfgSerializeVariable
{

public:

    inline SerializeSimpleStringVar
    (
        SimpleStringVar& baseVar,
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
        SimpleString tmp;
        OutputSimpleString output(tmp);
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

    SimpleStringVar& baseVar;

    const CharArray nameDesc;
    const CharArray comment;
    const CharArray blockComment;

};

//================================================================
//
// NumericVar<Type>::serialize
//
//================================================================

inline bool SimpleStringVar::serialize(const CfgSerializeKit& kit, const CharArray& name, const CharArray& comment, const CharArray& blockComment)
{
    auto oldValue = value;
    SerializeSimpleStringVar serializeVar(*this, name, comment, blockComment);
    kit.visitor(kit.scope, serializeVar);
    return allv(oldValue == value);
}
