#if defined(_WIN32)

#include "binaryFileWin32.h"

#include <windows.h>

#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "userOutput/errorLogEx.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "compileTools/classContext.h"
#include "formatting/formatModifiers.h"
#include "formattedOutput/requireMsg.h"
#include "numbers/int/intType.h"

//================================================================
//
// Win32Error
//
//================================================================

class Win32Error
{

public:

    operator DWORD () const {return error;}
    CLASS_CONTEXT(Win32Error, ((DWORD, error)))

};

//----------------------------------------------------------------

template <>
void formatOutput(const Win32Error& value, FormatOutputStream& outputStream)
{
    DWORD err = value;

    LPTSTR formatStr = 0;
    REMEMBER_CLEANUP1(DEBUG_BREAK_CHECK(LocalFree(formatStr) == 0), LPTSTR, formatStr);

    DWORD formatResult = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, value, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &formatStr, 0, NULL);

    ////

    size_t formatLen = strlen(formatStr);

    while (formatLen != 0 && (formatStr[formatLen-1] == '\r' || formatStr[formatLen-1] == '\n'))
        --formatLen;

    ////

    if (formatResult && formatStr)
        outputStream.write(CharArray(formatStr, formatLen));
    else
    {
        outputStream.write(STR("Error 0x"));
        outputStream.write(hex(uint32(value), 8));
    }
}

//================================================================
//
// getLastError
//
//================================================================

inline Win32Error getLastError() {return Win32Error(GetLastError());}

//================================================================
//
// BinaryFileWin32::close
//
//================================================================

void BinaryFileWin32::close()
{
    if (handle != 0)
    {
        DEBUG_BREAK_CHECK(CloseHandle(handle) != 0);
        handle = 0;
        currentFilename.clear();
        currentSize = 0;
        currentPosition = 0;
    }
}

//================================================================
//
// BinaryFileWin32::open
//
//================================================================

stdbool BinaryFileWin32::open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit))
{
    stdBegin;

    ////

    REQUIRE(filename.size >= 0);
    SimpleString newFilename(filename.ptr, filename.size);

    REQUIRE_TRACE0(newFilename.ok(), STR("Not enough memory"));

    ////

    HANDLE newHandle = CreateFile
    (
        newFilename.cstr(),
        !writeAccess ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE),
        !writeAccess ? FILE_SHARE_READ : FILE_SHARE_READ,
        NULL,
        createIfNotExists ? OPEN_ALWAYS : OPEN_EXISTING,
        0,
        NULL
    );

    REQUIRE_MSG2(newHandle != INVALID_HANDLE_VALUE, STR("Cannot open file %0: %1"), filename, getLastError());

    ////

    LARGE_INTEGER tmpSize;
    REQUIRE(GetFileSizeEx(newHandle, &tmpSize) != 0);
    REQUIRE(tmpSize.QuadPart >= 0);
    uint64 newSize = tmpSize.QuadPart;

    ////

    close();
    handle = newHandle;
    exchange(currentFilename, newFilename);
    currentSize = newSize;
    currentPosition = 0;

    stdEnd;
}

//================================================================
//
// BinaryFileWin32::truncate
//
//================================================================

stdbool BinaryFileWin32::truncate(stdPars(FileDiagKit))
{
    stdBegin;

    REQUIRE(handle);

    BOOL result = SetEndOfFile(handle);
    REQUIRE_MSG3(result != 0, STR("Cannot truncate file %0 at offset %1: %2"), currentFilename, currentPosition, getLastError());

    currentSize = currentPosition;

    stdEnd;
}

//================================================================
//
// BinaryFileWin32::setPosition
//
//================================================================

stdbool BinaryFileWin32::setPosition(uint64 pos, stdPars(FileDiagKit))
{
    stdBegin;

    REQUIRE(handle);
    REQUIRE(pos <= currentSize);

    ////

    LARGE_INTEGER largePos;
    largePos.QuadPart = pos;

    BOOL result = SetFilePointerEx(handle, largePos, 0, FILE_BEGIN);
    REQUIRE_MSG3(result != 0, STR("Cannot seek to offset %0 in file %1: %2"), pos, currentFilename, getLastError());

    ////

    currentPosition = pos;

    stdEnd;
}

//================================================================
//
// BinaryFileWin32::read
//
//================================================================

stdbool BinaryFileWin32::read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    stdBegin;

    REQUIRE(handle != 0);

    ////

    REQUIRE(dataSize <= typeMax<DWORD>());
    REQUIRE(dataSize <= currentSize - currentPosition);

    ////

    REMEMBER_CLEANUP2_EX
    (
        restorePositionCleanup,
        {
            LARGE_INTEGER restorePos;
            restorePos.QuadPart = currentPosition;
            DEBUG_BREAK_CHECK(SetFilePointerEx(handle, restorePos, 0, FILE_BEGIN) != 0);
        },
        HANDLE, handle, uint64, currentPosition
    );

    ////

    DWORD actualBytes = 0;

    DWORD dataSizeOS = 0;
    REQUIRE(convertExact(dataSize, dataSizeOS));
    BOOL result = ReadFile(handle, dataPtr, dataSizeOS, &actualBytes, NULL);

    CharArray errorMsg = STR("Cannot read %0 bytes at offset %1 from file %2: %3");
    REQUIRE_MSG4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
    REQUIRE_MSG4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, Win32Error(ERROR_HANDLE_EOF));

    ////

    currentPosition += actualBytes;

    restorePositionCleanup.cancel();

    stdEnd;
}

//================================================================
//
// BinaryFileWin32::write
//
//================================================================

stdbool BinaryFileWin32::write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    stdBegin;

    REQUIRE(handle != 0);

    ////

    REQUIRE(dataSize <= typeMax<DWORD>());
    REQUIRE(dataSize <= typeMax<uint64>() - currentSize);

    ////

    REMEMBER_CLEANUP2_EX
    (
        restorePositionCleanup,
        {
            LARGE_INTEGER restorePos;
            restorePos.QuadPart = currentPosition;
            DEBUG_BREAK_CHECK(SetFilePointerEx(handle, restorePos, 0, FILE_BEGIN) != 0);
        },
        HANDLE, handle, uint64, currentPosition
    );

    ////

    DWORD actualBytes = 0;
    DWORD dataSizeOS = 0;
    REQUIRE(convertExact(dataSize, dataSizeOS));
    BOOL result = WriteFile(handle, dataPtr, dataSizeOS, &actualBytes, NULL);

    CharArray errorMsg = STR("Cannot write %0 bytes at offset %1 to file %2: %3");
    REQUIRE_MSG4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
    REQUIRE_MSG4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, Win32Error(ERROR_HANDLE_EOF));

    ////

    currentPosition += actualBytes;
    currentSize = maxv(currentSize, currentPosition);

    restorePositionCleanup.cancel();

    stdEnd;
}

//----------------------------------------------------------------

#endif
