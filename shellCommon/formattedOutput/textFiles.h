#pragma once

#include <fstream>
#include <sstream>

#include "formattedOutput/messageFormatterStdio.h"
#include "stdFunc/stdFunc.h"
#include "stlString/stlString.h"
#include "userOutput/errorLogEx.h"
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

    stdbool open(const CharType* filename, stdPars(TextFileKit))
    {
        stream.close();
        stream.clear();

        stream.open(filename);
        REQUIRE_TRACE1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        returnTrue;
    }

    bool getLine(std::basic_string<Type>& s, stdPars(TextFileKit))
    {
        getline(stream, s);

        bool ok = !!stream;

        if_not (ok)
            CHECK_TRACE1(stream.eof(), STR("Cannot read file %0"), openedFilename);

        ////

        if (s.size() && s[s.size() - 1] == '\r')
            s = s.substr(0, s.size() - 1);

        ////

        return ok;
    }

    bool eof() const {return stream.eof();}

    stdbool readEntireFileToString(StlString& result, stdPars(TextFileKit))
    {
        std::basic_stringstream<CharType> strStream;
        strStream << stream.rdbuf();
        result = strStream.str();

        bool ok = !!stream;

        if_not (ok)
        {
            CHECK_TRACE1(stream.eof(), STR("Cannot read file %0"), openedFilename);
            returnFalse;
        }

        returnTrue;
    }

private:

    StlString openedFilename;
    std::basic_ifstream<Type> stream;

};

//================================================================
//
// InputTextFileUnicodeToAscii
//
// Reads 16-bit unicode file and translates to ASCII characters 0..7F.
//
//================================================================

#if 0

class InputTextFileUnicodeToAscii
{

public:

    stdbool open(StlString filename, stdPars(TextFileKit))
    {
        using namespace std;

        stream.close();
        stream.clear();

        ////

        stream.open(filename.c_str());
        REQUIRE_TRACE1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        returnTrue;
    }

    stdbool getLine(StlString& result, stdPars(TextFileKit))
    {
        std::basic_string<char> wstr;

        getline(stream, wstr);

        bool ok = !!stream;

        if_not (ok)
            CHECK_TRACE1(stream.eof(), STR("Cannot read file %0"), openedFilename);

        ////

        COMPILE_ASSERT(sizeof(wchar_t) == 2);
        const wchar_t* inputPtr = (wchar_t*) wstr.c_str();
        size_t inputSize = wstr.size() / 2; // round down

        if (inputSize >= 1 && inputPtr[inputSize-1] == '\r')
            --inputSize;

        ////

        size_t validCount = 0;

        for_count (k, inputSize)
            if (uint32(inputPtr[k]) < 0x80)
                ++validCount;

        ////

        result.resize(validCount);
        StlString::iterator outputPtr = result.begin();

        for_count (k, inputSize)
        {
            uint32 value = inputPtr[k];

            if (value < 0x80)
                *outputPtr++ = CharType(value);
        }

        ////

        return ok;
    }

    bool eof() const {return stream.eof();}

private:

    StlString openedFilename;
    std::basic_ifstream<char> stream;

};

#endif

//================================================================
//
// OutputTextFile
//
//================================================================

class OutputTextFile
{

public:

    stdbool open(const CharType* filename, stdPars(TextFileKit))
    {
        close();

        stream.open(filename);
        REQUIRE_TRACE1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        returnTrue;
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

    stdbool flushAndClose(stdPars(TextFileKit))
    {
        stream.flush();
        bool ok = !!stream;
        stream.close();
        stream.clear();
        REQUIRE_TRACE1(ok, STR("Cannot write to file %0"), openedFilename);
        returnTrue;
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

        file.write(formatter.data(), formatter.size());

        errorExit.cancel();
        return true;
    }

    bool clear()
        {return true;}

    bool update()
        {return true;}

    bool isThreadProtected() const
        {return false;}

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
