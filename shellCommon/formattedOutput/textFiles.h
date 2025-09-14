#pragma once

#include <fstream>
#include <sstream>

#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "stdFunc/stdFunc.h"
#include "stlString/stlString.h"
#include "userOutput/printMsgTrace.h"
#include "userOutput/msgLog.h"
#include "userOutput/printMsg.h"
#include "userOutput/diagnosticKit.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// TextFileKit
//
//================================================================

using TextFileKit = DiagnosticKit;

//================================================================
//
// InputTextFile
//
//================================================================

template <typename Type>
class InputTextFile
{

public:

    void open(const CharType* filename, stdPars(TextFileKit))
    {
        stream.close();
        stream.clear();

        stream.open(filename);
        REQUIRE_TRACE1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;
    }

    bool getLine(std::basic_string<Type>& s, stdPars(TextFileKit))
    {
        getline(stream, s);

        bool ok = !!stream;

        if_not (ok || stream.eof())
            errorBlock(printMsgTrace(STR("Cannot read file %0"), openedFilename, msgErr, stdPassNc));

        ////

        if (s.size() && s[s.size() - 1] == '\r')
            s = s.substr(0, s.size() - 1);

        ////

        return ok;
    }

    bool eof() const {return stream.eof();}

    void readEntireFileToString(StlString& result, stdPars(TextFileKit))
    {
        std::basic_stringstream<CharType> strStream;
        strStream << stream.rdbuf();
        result = strStream.str();

        bool ok = !!stream;
        REQUIRE_TRACE1(ok || stream.eof(), STR("Cannot read file %0"), openedFilename);
    }

private:

    StlString openedFilename;
    std::basic_ifstream<Type> stream;

};

//================================================================
//
// OutputTextFile
//
//================================================================

class OutputTextFile
{

public:

    void open(const CharType* filename, stdPars(TextFileKit))
    {
        close();

        stream.open(filename);
        REQUIRE_TRACE1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;
    }

    void close()
    {
        stream.close();
        stream.clear();
        openedFilename.clear();
    }

    bool valid() const
    {
        return !!stream;
    }

    void invalidate()
    {
        stream.setstate(std::ios::failbit);
    }

    void flushAndClose(stdPars(TextFileKit))
    {
        stream.flush();
        bool ok = !!stream;
        stream.close();
        stream.clear();
        REQUIRE_TRACE1(ok, STR("Cannot write to file %0"), openedFilename);
    }

    void write(const CharType* ptr, size_t size)
    {
        stream.write(ptr, size);
    }

private:

    StlString openedFilename;
    std::basic_ofstream<CharType> stream;

};

//================================================================
//
// MsgLogToTextFile
//
//================================================================

class MsgLogToTextFile : public MsgLog
{

public:

    MsgLogToTextFile(OutputTextFile& file, MessageFormatter& formatter)
        : file(file), formatter(formatter) {}

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind)
    {
        REMEMBER_CLEANUP_EX(errorExit, file.invalidate());

        formatter.clear();
        v.func(v.value, formatter);
        formatter.write(CT("\n"), 1);
        ensure(formatter.valid());

        file.write(formatter.cstr(), formatter.size());

        errorExit.cancel();
        return true;
    }

    bool clear()
        {return true;}

    bool update()
        {return true;}

    virtual void lock()
        {}

    virtual void unlock()
        {}

private:

    OutputTextFile& file;
    MessageFormatter& formatter;

};

//----------------------------------------------------------------

inline auto getLog(OutputTextFile& file, const MessageFormatterKit& kit)
{
    return MsgLogToTextFile{file, kit.formatter};
}
