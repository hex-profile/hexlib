#if defined(_WIN32)

#include "errorWin32.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "formatting/formatStream.h"
#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "formatting/formatModifiers.h"

//================================================================
//
// ErrorWin32
//
//================================================================

template <>
void formatOutput(const ErrorWin32& value, FormatOutputStream& outputStream)
{
    LPTSTR formatStr = 0;
    REMEMBER_CLEANUP_DBGBRK_CHECK(LocalFree(formatStr) == 0);

    DWORD formatResult = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, value.get(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &formatStr, 0, NULL);

    ////

    size_t formatLen = strlen(formatStr);

    while (formatLen != 0 && (formatStr[formatLen-1] == '\r' || formatStr[formatLen-1] == '\n'))
        --formatLen;

    ////

    outputStream << STR("Error ") << value.get();

    if (formatResult && formatStr)
        outputStream << STR(". ") << charArray(formatStr, formatLen);
}

//----------------------------------------------------------------

#endif
