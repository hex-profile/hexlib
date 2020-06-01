//#if defined(__linux__)
//
//#include "binaryFileLinux.h"
//
//#include "compileTools/classContext.h"
//#include "errorLog/debugBreak.h"
//#include "errorLog/errorLog.h"
//#include "formatting/formatModifiers.h"
//#include "numbers/int/intType.h"
//#include "storage/rememberCleanup.h"
//#include "userOutput/errorLogEx.h"
//#include "userOutput/errorLogEx.h"
//#include "userOutput/printMsg.h"
//
////================================================================
////
//// getLastError
////
////================================================================
//
//inline ErrorLinux getLastError() {return ErrorLinux(GetLastError());}
//
////================================================================
////
//// BinaryFileLinux::close
////
////================================================================
//
//void BinaryFileLinux::close()
//{
//    if (handle != 0)
//    {
//        DEBUG_BREAK_CHECK(CloseHandle(handle) != 0);
//        handle = 0;
//        currentFilename.clear();
//        currentSize = 0;
//        currentPosition = 0;
//    }
//}
//
////================================================================
////
//// BinaryFileLinux::open
////
////================================================================
//
//stdbool BinaryFileLinux::open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit))
//{
//    REQUIRE(filename.size >= 0);
//    SimpleString newFilename(filename.ptr, filename.size);
//
//    REQUIRE_TRACE0(def(newFilename), STR("Not enough memory"));
//
//    ////
//
//    HANDLE newHandle = CreateFile
//    (
//        newFilename.cstr(),
//        !writeAccess ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE),
//        !writeAccess ? FILE_SHARE_READ : FILE_SHARE_READ,
//        NULL,
//        createIfNotExists ? OPEN_ALWAYS : OPEN_EXISTING,
//        0,
//        NULL
//    );
//
//    REQUIRE_TRACE2(newHandle != INVALID_HANDLE_VALUE, STR("Cannot open file %0: %1"), filename, getLastError());
//
//    ////
//
//    LARGE_INTEGER tmpSize;
//    REQUIRE(GetFileSizeEx(newHandle, &tmpSize) != 0);
//    REQUIRE(tmpSize.QuadPart >= 0);
//    uint64 newSize = tmpSize.QuadPart;
//
//    ////
//
//    close();
//    handle = newHandle;
//    exchange(currentFilename, newFilename);
//    currentSize = newSize;
//    currentPosition = 0;
//
//    returnTrue;
//}
//
////================================================================
////
//// BinaryFileLinux::truncate
////
////================================================================
//
//stdbool BinaryFileLinux::truncate(stdPars(FileDiagKit))
//{
//    REQUIRE(handle);
//
//    BOOL result = SetEndOfFile(handle);
//    REQUIRE_TRACE3(result != 0, STR("Cannot truncate file %0 at offset %1: %2"), currentFilename, currentPosition, getLastError());
//
//    currentSize = currentPosition;
//
//    returnTrue;
//}
//
////================================================================
////
//// BinaryFileLinux::setPosition
////
////================================================================
//
//stdbool BinaryFileLinux::setPosition(uint64 pos, stdPars(FileDiagKit))
//{
//    REQUIRE(handle);
//    REQUIRE(pos <= currentSize);
//
//    ////
//
//    LARGE_INTEGER largePos;
//    largePos.QuadPart = pos;
//
//    BOOL result = SetFilePointerEx(handle, largePos, 0, FILE_BEGIN);
//    REQUIRE_TRACE3(result != 0, STR("Cannot seek to offset %0 in file %1: %2"), pos, currentFilename, getLastError());
//
//    ////
//
//    currentPosition = pos;
//
//    returnTrue;
//}
//
////================================================================
////
//// BinaryFileLinux::read
////
////================================================================
//
//stdbool BinaryFileLinux::read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
//{
//    REQUIRE(handle != 0);
//
//    ////
//
//    REQUIRE(dataSize <= typeMax<DWORD>());
//    REQUIRE(dataSize <= currentSize - currentPosition);
//
//    ////
//
//    REMEMBER_CLEANUP2_EX
//    (
//        restorePositionCleanup,
//        {
//            LARGE_INTEGER restorePos;
//            restorePos.QuadPart = currentPosition;
//            DEBUG_BREAK_CHECK(SetFilePointerEx(handle, restorePos, 0, FILE_BEGIN) != 0);
//        },
//        HANDLE, handle, uint64, currentPosition
//    );
//
//    ////
//
//    DWORD actualBytes = 0;
//
//    DWORD dataSizeOS = 0;
//    REQUIRE(convertExact(dataSize, dataSizeOS));
//    BOOL result = ReadFile(handle, dataPtr, dataSizeOS, &actualBytes, NULL);
//
//    CharArray errorMsg = STR("Cannot read %0 bytes at offset %1 from file %2: %3");
//    REQUIRE_TRACE4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
//    REQUIRE_TRACE4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, ErrorLinux(ERROR_HANDLE_EOF));
//
//    ////
//
//    currentPosition += actualBytes;
//
//    restorePositionCleanup.cancel();
//
//    returnTrue;
//}
//
////================================================================
////
//// BinaryFileLinux::write
////
////================================================================
//
//stdbool BinaryFileLinux::write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
//{
//    REQUIRE(handle != 0);
//
//    ////
//
//    REQUIRE(dataSize <= typeMax<DWORD>());
//    REQUIRE(dataSize <= typeMax<uint64>() - currentSize);
//
//    ////
//
//    REMEMBER_CLEANUP2_EX
//    (
//        restorePositionCleanup,
//        {
//            LARGE_INTEGER restorePos;
//            restorePos.QuadPart = currentPosition;
//            DEBUG_BREAK_CHECK(SetFilePointerEx(handle, restorePos, 0, FILE_BEGIN) != 0);
//        },
//        HANDLE, handle, uint64, currentPosition
//    );
//
//    ////
//
//    DWORD actualBytes = 0;
//    DWORD dataSizeOS = 0;
//    REQUIRE(convertExact(dataSize, dataSizeOS));
//    BOOL result = WriteFile(handle, dataPtr, dataSizeOS, &actualBytes, NULL);
//
//    CharArray errorMsg = STR("Cannot write %0 bytes at offset %1 to file %2: %3");
//    REQUIRE_TRACE4(result != 0, errorMsg, dataSize, currentPosition, currentFilename, getLastError());
//    REQUIRE_TRACE4(actualBytes == dataSize, errorMsg, dataSize, currentPosition, currentFilename, ErrorLinux(ERROR_HANDLE_EOF));
//
//    ////
//
//    currentPosition += actualBytes;
//    currentSize = maxv(currentSize, currentPosition);
//
//    restorePositionCleanup.cancel();
//
//    returnTrue;
//}
//
////----------------------------------------------------------------
//
//#endif