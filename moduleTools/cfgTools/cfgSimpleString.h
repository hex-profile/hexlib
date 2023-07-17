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

    template <typename AnyString>
    explicit SimpleStringVarEx(const AnyString& defaultValue)
    {
        this->value = defaultValue;
        this->defaultValue = defaultValue;
    }

public:

    template <typename AnyString>
    inline void setDefaultValue(const AnyString& defaultValue)
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

            synced = false;
        }
    }

public:

    inline void clear()
    {
        if_not (value.valid() && value.size() == 0)
        {
            value.clear();
            synced = false;
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

    bool synced = false;

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

    bool synced() const
        {return baseVar.synced;}

    void setSynced(bool value) const
        {baseVar.synced = value;}

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
        ensure(s.readAll(output));
        ensure(def(tmp));
        baseVar = tmp;
        return true;
    }

    bool getTextComment(CfgWriteStream& s) const
    {
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
    auto oldValue{value};
    SerializeSimpleStringVar<Type> serializeVar(*this, name, comment, blockComment);
    kit.visitVar(serializeVar);
    return allv(oldValue == value);
}

//================================================================
//
// SimpleStringVar
// SimpleStringVarChar
//
//================================================================

using SimpleStringVar = SimpleStringVarEx<CharType>;

using SimpleStringVarChar = SimpleStringVarEx<char>;
