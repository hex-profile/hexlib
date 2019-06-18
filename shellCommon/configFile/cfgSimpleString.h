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
        bool ok = str.ok();

        if_not (str.ok())
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

    SimpleStringVar()
        {}

    SimpleStringVar(const SimpleString& value)
        {operator =(value);}

    SimpleStringVar(const CharType* value)
        {operator =(SimpleString(value));}

public:

    inline void operator =(const SimpleString& X)
    {
        if (value != X)
        {
            value = X;

            if_not (value.ok())
                value.clear();

            changed = true;
        }
    }

public:

    void clear()
    {
        value.clear();
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

private:

    friend class SerializeSimpleString;

    SimpleString value;
    bool changed = false;

};

//================================================================
//
// CfgSimpleString
//
//================================================================

class SerializeSimpleString : public CfgSerializeVariable
{

public:

    inline SerializeSimpleString
    (
        SimpleStringVar& baseVar,
        const CharArray& nameDesc,
        const CharArray& comment = STR("")
    )
        :
        baseVar(baseVar),
        nameDesc(nameDesc),
        comment(comment)
    {
    }

private:

    bool changed() const
        {return baseVar.changed;}

    void clearChanged() const
        {baseVar.changed = false;}

    void resetValue() const // ```
        {baseVar.clear();}

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
        ensure(tmp.ok());
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
        {return true;}

private:

    SimpleStringVar& baseVar;

    CharArray const nameDesc;
    CharArray const comment;

};
