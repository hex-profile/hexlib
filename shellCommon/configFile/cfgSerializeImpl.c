#include "cfgSerializeImpl.h"

#include <float.h>
#include <sstream>
#include <iomanip>

#include "cfg/cfgInterface.h"

namespace cfgVarsImpl {

using namespace std;

//================================================================
//
// Cfgvar set implementation using STL for format I/O
// and loading/storing to StringEnv interface.
//
//================================================================

//================================================================
//
// String
//
//================================================================

using String = basic_string<CharType>;

//================================================================
//
// WriteStreamStlThunk
//
//----------------------------------------------------------------
//
// CfgWriteStream implementation using STL
//
//================================================================

class WriteStreamStlThunk : public CfgWriteStream
{

private:

    basic_ostringstream<CharType> stream;

    //----------------------------------------------------------------

public:

    inline bool ok() {return !!stream;}
    inline String str() {return stream.str();}

    //----------------------------------------------------------------

public:

    bool writeChars(const CharType* array, size_t size)
    {
        try {stream.write(array, size); ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    //----------------------------------------------------------------

public:

    bool writeCstr(const CharType* cstr)
    {
        try {stream << cstr; ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    //----------------------------------------------------------------

public:

    #define TMP_MACRO(Type, o) \
        \
        bool write(Type value) \
        { \
            try {stream << value; ensure(!!stream);} \
            catch (const exception&) {return false;} \
            return true; \
        }

    BUILTIN_INT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

public:

    bool write(float value, int32 precision)
    {
        if (precision == 0) precision = FLT_DIG+1;
        try {stream << setprecision(precision) << value; ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    bool write(double value, int32 precision)
    {
        if (precision == 0) precision = DBL_DIG+1;
        try {stream << setprecision(precision) << value; ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    //----------------------------------------------------------------

public:

    void clear()
        {stream.clear();}

};

//================================================================
//
// ReadStreamStlThunk
//
//----------------------------------------------------------------
//
// CfgReadStream implementation using STL
//
//================================================================

class ReadStreamStlThunk : public CfgReadStream
{

private:

    basic_istringstream<CharType> stream;

    //----------------------------------------------------------------

public:

    inline ReadStreamStlThunk(const CharType* str)
        : stream(str) {}

    //----------------------------------------------------------------

public:

    bool unreadChar()
    {
        try {stream.unget(); ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    //----------------------------------------------------------------

public:

    bool readChars(CharType* result, size_t size)
    {
        try {stream.read(result, size); ensure(!!stream);}
        catch (const exception&) {return false;}
        return true;
    }

    //----------------------------------------------------------------

    #define TMP_MACRO(Type, o) \
        \
        bool read(Type& result) \
        { \
            try {stream >> result; ensure(!!stream);} \
            catch (const exception&) {return false;} \
            return true; \
        }

    BUILTIN_INT_FOREACH(TMP_MACRO, o)
    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    //----------------------------------------------------------------

    bool readString(CfgOutputString& result)
    {

        try
        {
            String s;
            getline(stream, s);

            ensure(stream || stream.eof());
            ensure(result.addBuf(s.c_str(), s.size()));

        }
        catch (const exception&) {return false;}

        return true;
    }

};

//================================================================
//
// GetNameToContainer
//
//================================================================

struct GetNameToContainer : public CfgOutputString
{

public:

    bool addBuf(const CharType* bufArray, size_t bufSize)
    {

        try
        {
            NamePart tmp(String(bufArray, bufSize));
            container.push_front(tmp);
        }
        catch (const exception&) {return false;}

        return true;
    }

public:

    inline GetNameToContainer(NameContainer& container)
        : container(container) {}

private:

    NameContainer& container;

};

//================================================================
//
// getVarNamePath
//
//================================================================

bool getVarNamePath(const CfgNamespace* scope, const CfgSerializeVariable& var, NameContainer& result)
{
    GetNameToContainer getName(result);

    // Get main part
    ensure(var.getName(getName));

    // Get space scope
    for (const CfgNamespace* p = scope; p != 0; p = p->prev)
        ensure(getName.addStr(charArrayFromPtr(p->desc)));

    return true;
}

//================================================================
//
// loadVar
//
// Can throw exceptions.
//
//================================================================

bool loadVar(const CfgNamespace* scope, const CfgSerializeVariable& var, StringEnv& stringEnv)
{
    //
    // Get the variable name.
    //

    NameContainer name;
    ensure(getVarNamePath(scope, var, name));

    //
    // Try to get string value.
    //

    String varValue;
    String varComment;
    String varBlockComment;

    ensure(stringEnv.get(name, varValue, varComment, varBlockComment));

    //
    // And to set variable value.
    //

    ReadStreamStlThunk tmp(varValue.c_str()); // [D184B846]

    ensure(var.setTextValue(tmp));

    return true;
}

//================================================================
//
// LoadVar
//
//================================================================

class LoadVar : public CfgVisitor
{

    StringEnv& stringEnv;

public:

    LoadVar(StringEnv& stringEnv)
        : stringEnv(stringEnv) {}

    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
    {
        try {loadVar(scope, var, stringEnv);}
        catch (const exception&) {}
    }

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
    {
    }

};

//================================================================
//
// saveVar
//
// Can throw exceptions.
//
//================================================================

bool saveVar(const CfgNamespace* scope, const CfgSerializeVariable& var, StringEnv& stringEnv)
{

    //
    // Get the variable name.
    //

    NameContainer name;
    ensure(getVarNamePath(scope, var, name));

    //
    // Get text representation of the variable's value
    //

    WriteStreamStlThunk value;

    if_not (var.getTextValue(value) && value.ok())
        value.clear();

    //
    // Get comments
    //

    WriteStreamStlThunk comment;

    if_not (var.getTextComment(comment) && comment.ok())
        comment.clear();

    ////

    WriteStreamStlThunk blockComment;

    if_not (var.getBlockComment(blockComment) && blockComment.ok())
        blockComment.clear();

    //
    // Set name and comment
    //

    String oldValue;
    String oldValueComment;
    String oldBlockComment;
    stringEnv.get(name, oldValue, oldValueComment, oldBlockComment); // ignore error

    String commentStr = comment.str();
    String blockCommentStr = blockComment.str();

    ensure
    (
        stringEnv.set
        (
            name,
            value.str(),
            !commentStr.empty() ? commentStr : oldValueComment, // if empty, take the old one
            !blockCommentStr.empty() ? blockCommentStr : oldBlockComment // if empty, take the old one
        )
    );

    return true;
}

//================================================================
//
// SaveVar
//
//================================================================

class SaveVar : public CfgVisitor
{

    StringEnv& stringEnv;

public:

    SaveVar(StringEnv& stringEnv)
        : stringEnv(stringEnv) {}

    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
    {
        try {saveVar(scope, var, stringEnv);}
        catch (const exception&) {}
    }

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
        {}

};

//================================================================
//
// saveVarsToStringEnv
//
//================================================================

void saveVarsToStringEnv(CfgSerialization& serialization, const CfgNamespace* scope, StringEnv& stringEnv)
{
    SaveVar saveVar(stringEnv);
    CfgSerializeKit kit(saveVar, scope);
    serialization.serialize(kit);
}

//================================================================
//
// loadVarsFromStringEnv
//
//================================================================

void loadVarsFromStringEnv(CfgSerialization& serialization, const CfgNamespace* scope, StringEnv& stringEnv)
{
    LoadVar loadVar(stringEnv);
    CfgSerializeKit kit(loadVar, scope);
    serialization.serialize(kit);
}

//----------------------------------------------------------------

}
