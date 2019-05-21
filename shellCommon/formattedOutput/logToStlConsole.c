#include "logToStlConsole.h"

#include <sstream>
#include <iostream>

#if defined(_WIN32)
    #include <windows.h>
#endif

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStl.h"
#include "stlString/stlString.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// LogToStlConsole::addMsg
//
//================================================================

bool LogToStlConsole::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    using namespace std;

    try
    {
        basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        v.func(v.value, formatToStream);
        ensure(formatToStream.isOk());

        stringStream << endl;
        ensure(!!stringStream);

        {
            lock_guard<decltype(mutex)> guard(mutex);

            StlString str = stringStream.rdbuf()->str();

            ostream& screenStream = (msgKind >= msgWarn) ? cerr : cout;
            screenStream << str;

            screenStream.flush();

        #if defined(_WIN32)

            if (useDebugOutput)
                OutputDebugString(str.c_str());

        #endif

        }

    }
    catch (const exception&) {return false;}

    return true;
}
