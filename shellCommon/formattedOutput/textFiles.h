#pragma once

#include <fstream>
#include <sstream>

#include "formattedOutput/formatStreamStdio.h"
#include "stdFunc/stdFunc.h"
#include "stlString/stlString.h"
#include "userOutput/errorLogEx.h"
#include "userOutput/msgLog.h"
#include "userOutput/printMsg.h"
#include "userOutput/diagnosticKit.h"

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

class OutputTextFile : public MsgLog
{

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind)
    {
        using namespace std;

        try
        {
            constexpr size_t bufferSize = 1024;
            CharType bufferArray[bufferSize];
            FormatStreamStdioThunk formatter{bufferArray, bufferSize};

            v.func(v.value, formatter);
            formatter.write(CT("\n"), 1);
            ensure(formatter.valid());

            stream << formatter.data();
            ensure(!!stream);
        }
        catch (const exception&) {return false;}

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

    stdbool flush(stdPars(TextFileKit))
    {
        stream.flush();
        REQUIRE_TRACE1(!!stream, STR("Cannot write to file %0"), openedFilename);
        returnTrue;
    }

    stdbool flushClose(stdPars(TextFileKit))
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
