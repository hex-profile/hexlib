#if defined(_WIN32)

#include "binaryFileWin32.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgTrace.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "compileTools/classContext.h"
#include "formatting/formatModifiers.h"
#include "userOutput/printMsgTrace.h"
#include "numbers/int/intType.h"
#include "osErrors/errorWin32.h"

//================================================================
//
// getLastError
//
//================================================================

inline ErrorWin32 getLastError() {return ErrorWin32(GetLastError());}

//================================================================
//
// BinaryFileWin32::close
//
//================================================================

void BinaryFileWin32::close()
{
    if (currentHandle != 0)
    {
        DEBUG_BREAK_CHECK(CloseHandle(currentHandle) != 0);
        currentHandle = 0;
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
    close();

    ////

    REQUIRE(filename.size >= 0);
    SimpleString newFilename(filename.ptr, filename.size);

    REQUIRE_TRACE(def(newFilename), STR("Not enough memory"));

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

    REQUIRE_TRACE2(newHandle != INVALID_HANDLE_VALUE, STR("Cannot open file %0: %1"), filename, getLastError());

    REMEMBER_CLEANUP_EX(newHandleCleanup, DEBUG_BREAK_CHECK(CloseHandle(newHandle) != 0));

    ////

    LARGE_INTEGER tmpSize;
    REQUIRE(GetFileSizeEx(newHandle, &tmpSize) != 0);
    REQUIRE(tmpSize.QuadPart >= 0);
    uint64 newSize = tmpSize.QuadPart;

    ////

    newHandleCleanup.cancel();

    currentHandle = newHandle;
    exchange(currentFilename, newFilename);
    currentSize = newSize;
    currentPosition = 0;

    returnTrue;
}

//================================================================
//
// BinaryFileWin32::truncate
//
//================================================================

stdbool BinaryFileWin32::truncate(stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != 0);

    BOOL result = SetEndOfFile(currentHandle);
    REQUIRE_TRACE3(result != 0, STR("Cannot truncate file %0 at offset %1: %2"), currentFilename, currentPosition, getLastError());

    currentSize = currentPosition;

    returnTrue;
}

//================================================================
//
// BinaryFileWin32::setPosition
//
//================================================================

stdbool BinaryFileWin32::setPosition(uint64 pos, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != 0);
    REQUIRE(pos <= currentSize);

    ////

    LARGE_INTEGER largePos;
    largePos.QuadPart = pos;

    BOOL result = SetFilePointerEx(currentHandle, largePos, 0, FILE_BEGIN);
    REQUIRE_TRACE3(result != 0, STR("Cannot seek to offset %0 in file %1: %2"), pos, currentFilename, getLastError());

    ////

    currentPosition = pos;

    returnTrue;
}

//================================================================
//
// BinaryFileWin32::read
//
//================================================================

stdbool BinaryFileWin32::read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != 0);

    ////

    REQUIRE(dataSize <= typeMax<DWORD>());
    REQUIRE(dataSize <= currentSize - currentPosition);

    ////

    REMEMBER_CLEANUP_EX
    (
        restorePositionCleanup,
        {
            LARGE_INTEGER restorePos;
            restorePos.QuadPart = currentPosition;
            DEBUG_BREAK_CHECK(SetFilePointerEx(currentHandle, restorePos, 0, FILE_BEGIN) != 0);
        }
    );

    ////

    DWORD actualBytes = 0;

    DWORD dataSizeOS = 0;
    REQUIRE(convertExact(dataSize, dataSizeOS));
    BOOL result = ReadFile(currentHandle, dataPtr, dataSizeOS, &actualBytes, NULL);

    CharArray errorMsg = STR("Cannot read %0 bytes at offset %1 from file %2: %3");
    REQUIRE_TRACE4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
    REQUIRE_TRACE4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, ErrorWin32(ERROR_HANDLE_EOF));

    ////

    currentPosition += actualBytes;

    restorePositionCleanup.cancel();

    returnTrue;
}

//================================================================
//
// BinaryFileWin32::write
//
//================================================================

stdbool BinaryFileWin32::write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != 0);

    ////

    REQUIRE(dataSize <= typeMax<DWORD>());
    REQUIRE(dataSize <= typeMax<uint64>() - currentSize);

    ////

    REMEMBER_CLEANUP_EX
    (
        restorePositionCleanup,
        {
            LARGE_INTEGER restorePos;
            restorePos.QuadPart = currentPosition;
            DEBUG_BREAK_CHECK(SetFilePointerEx(currentHandle, restorePos, 0, FILE_BEGIN) != 0);
        }
    );

    ////

    DWORD actualBytes = 0;
    DWORD dataSizeOS = 0;
    REQUIRE(convertExact(dataSize, dataSizeOS));
    BOOL result = WriteFile(currentHandle, dataPtr, dataSizeOS, &actualBytes, NULL);

    CharArray errorMsg = STR("Cannot write %0 bytes at offset %1 to file %2: %3");
    REQUIRE_TRACE4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
    REQUIRE_TRACE4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, ErrorWin32(ERROR_HANDLE_EOF));

    ////

    currentPosition += actualBytes;
    currentSize = maxv(currentSize, currentPosition);

    restorePositionCleanup.cancel();

    returnTrue;
}

//----------------------------------------------------------------

#endif
