#include "logToStdConsole.h"

#include <stdio.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStdio.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// logToStdConsole::addMsg
//
//================================================================

bool logToStdConsole::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    constexpr size_t bufferSize = 1024;
    CharType bufferArray[bufferSize];
    FormatStreamStdioThunk formatter{bufferArray, bufferSize};

    v.func(v.value, formatter);

    formatter.write(CT("\n"), 1);
    ensure(formatter.valid());

    {
        std::lock_guard<decltype(mutex)> guard(mutex);

        auto* screenStream = stdout;

        if (useStdErr && msgKind >= msgWarn) 
            screenStream = stderr;

        ensure(fwrite(formatter.data(), sizeof(CharType), formatter.size(), screenStream) == formatter.size());
        ensure(fflush(screenStream) == 0);

    #if defined(_WIN32)

        if (useDebugOutput)
            OutputDebugString(formatter.data());

    #endif

    }

    return true;
}
