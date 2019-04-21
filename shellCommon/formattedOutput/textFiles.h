#pragma once

#include <fstream>

#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLog.h"
#include "formattedOutput/formatStreamStl.h"
#include "stlString/stlString.h"
#include "userOutput/printMsg.h"
#include "formattedOutput/requireMsg.h"

//================================================================
//
// InputTextFile
//
//================================================================

template <typename Type>
class InputTextFile
{

public:

    stdbool open(StlString filename, stdPars(MsgLogKit))
    {
        stdBegin;

        stream.close();
        stream.clear();

        stream.open(filename.c_str());
        REQUIRE_MSG1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        stdEnd;
    }

    stdbool getLine(std::basic_string<Type>& s, stdPars(MsgLogKit))
    {
        getline(stream, s);

        bool ok = !!stream;

        if_not (ok)
            CHECK_EX(stream.eof(), printMsg(kit.msgLog, STR("Cannot read file %0"), openedFilename));

        return ok;
    }

    bool eof() const {return stream.eof();}

    stdbool readEntireFileToString(StlString& result, stdPars(MsgLogKit))
    {
        basic_stringstream<CharType> strStream;
        strStream << stream.rdbuf();
        result = strStream.str();

        bool ok = !!stream;

        if_not (ok)
            CHECK_EX(stream.eof(), printMsg(kit.msgLog, STR("Cannot read file %0"), openedFilename));

        return ok;
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

class InputTextFileUnicodeToAscii
{

public:

    stdbool open(StlString filename, stdPars(MsgLogKit))
    {
        stdBegin;

        using namespace std;

        stream.close();
        stream.clear();

        ////

        stream.open(filename.c_str());
        REQUIRE_MSG1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        stdEnd;
    }

    stdbool getLine(StlString& result, stdPars(MsgLogKit))
    {
        std::basic_string<char> wstr;

        getline(stream, wstr);

        bool ok = !!stream;

        if_not (ok)
            CHECK_EX(stream.eof(), printMsg(kit.msgLog, STR("Cannot read file %0"), openedFilename));

        ////

        COMPILE_ASSERT(sizeof(wchar_t) == 2);
        const wchar_t* inputPtr = (wchar_t*) wstr.c_str();
        size_t inputSize = wstr.size() / 2; // round down

        if (inputSize >= 1 && inputPtr[inputSize-1] == '\r')
            --inputSize;

        ////

        size_t validCount = 0;

        for (size_t k = 0; k < inputSize; ++k)
            if (uint32(inputPtr[k]) < 0x80)
                ++validCount;

        ////

        result.resize(validCount);
        StlString::iterator outputPtr = result.begin();

        for (size_t k = 0; k < inputSize; ++k)
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
            FormatStreamStlThunk formatToStream(stream);

            v.func(v.value, formatToStream);
            require(formatToStream.isOk());

            stream << endl;
            require(!!stream);
        }
        catch (const exception&) {require(false);}

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

    stdbool open(StlString filename, stdPars(MsgLogKit))
    {
        stdBegin;

        stream.close();
        stream.clear();

        stream.open(filename.c_str());
        REQUIRE_MSG1(!!stream, STR("Cannot open file %0"), filename);

        openedFilename = filename;

        stdEnd;
    }

    stdbool flush(stdPars(MsgLogKit))
    {
        stream.flush();
        REQUIRE_MSG1(!!stream, STR("Cannot write to file %0"), openedFilename);
        return true;
    }

    stdbool flushClose(stdPars(MsgLogKit))
    {
        stream.flush();
        bool ok = !!stream;
        stream.close();
        stream.clear();
        REQUIRE_MSG1(ok, STR("Cannot write to file %0"), openedFilename);
        return true;
    }

private:

    StlString openedFilename;
    std::basic_ofstream<CharType> stream;

};
