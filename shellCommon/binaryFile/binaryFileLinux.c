#if defined(__linux__)

#define _FILE_OFFSET_BITS 64

#include "binaryFileLinux.h"

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "formatting/formatModifiers.h"
#include "numbers/int/intType.h"
#include "osErrors/errorLinux.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsgTrace.h"
#include "userOutput/printMsg.h"
#include "numbers/int/intCompare.h"

//================================================================
//
// off_t
//
//================================================================

COMPILE_ASSERT(TYPE_BIT_COUNT(off_t) == 64);
COMPILE_ASSERT(TYPE_IS_SIGNED(off_t));

COMPILE_ASSERT(sizeof(off_t) <= sizeof(uint64));
COMPILE_ASSERT(TYPE_MAX(off_t) <= TYPE_MAX(uint64));

//================================================================
//
// getLastError
//
//================================================================

inline ErrorLinux getLastError() {return ErrorLinux(errno);}

//================================================================
//
// BinaryFileLinux::close
//
//================================================================

void BinaryFileLinux::close()
{
    if (currentHandle != 0)
    {
        DEBUG_BREAK_CHECK(::close(currentHandle) == 0);
        currentHandle = -1;
        currentFilename.clear();
        currentSize = 0;
        currentPosition = 0;
    }
}

//================================================================
//
// BinaryFileLinux::open
//
//================================================================

stdbool BinaryFileLinux::open(const CharArray& filename, bool writeAccess, bool createIfNotExists, stdPars(FileDiagKit))
{
    close();

    ////

    REQUIRE(filename.size >= 0);
    SimpleString newFilename(filename.ptr, filename.size);

    REQUIRE_TRACE(def(newFilename), STR("Not enough memory."));

    ////

    int flags = writeAccess ? O_RDWR : O_RDONLY;

    if (createIfNotExists)
        flags |= O_CREAT;

    ////

    int mode = 0644;

    ////

    int newHandle = ::open(newFilename.cstr(), flags, mode);

    REQUIRE_TRACE2(newHandle >= 0, STR("Cannot open file %0: %1."), filename, getLastError());

    REMEMBER_CLEANUP_EX(newHandleCleanup, DEBUG_BREAK_CHECK(::close(newHandle) == 0));

    ////

    auto newSize = lseek(newHandle, 0, SEEK_END);
    REQUIRE_TRACE2(newSize >= 0, STR("Cannot get file size %0: %1."), filename, getLastError());

    REQUIRE_TRACE2(lseek(newHandle, 0, SEEK_SET) >= 0, STR("Cannot seek to zero %0: %1."), filename, getLastError());

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
// BinaryFileLinux::truncate
//
//================================================================

stdbool BinaryFileLinux::truncate(stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != -1);

    auto result = ftruncate(currentHandle, currentPosition);
    REQUIRE_TRACE3(result == 0, STR("Cannot truncate file %0 at offset %1: %2"), currentFilename, currentPosition, getLastError());

    currentSize = currentPosition;

    returnTrue;
}

//================================================================
//
// BinaryFileLinux::setPosition
//
//================================================================

stdbool BinaryFileLinux::setPosition(uint64 pos, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != -1);
    REQUIRE(pos <= currentSize);

    ////

    REQUIRE(pos <= TYPE_MAX(off_t));
    auto result = lseek(currentHandle, pos, SEEK_SET);

    REQUIRE_TRACE3(intEqual(result, pos), STR("Cannot seek to offset %0 in file %1: %2"), pos, currentFilename, getLastError());

    ////

    currentPosition = pos;

    returnTrue;
}

//================================================================
//
// BinaryFileLinux::read
//
//================================================================

stdbool BinaryFileLinux::read(void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != -1);

    ////

    REQUIRE(dataSize <= currentSize - currentPosition);

    ////

    REMEMBER_CLEANUP_EX
    (
        restorePositionCleanup,
        {
            auto result = lseek(currentHandle, currentPosition, SEEK_SET);
            DEBUG_BREAK_CHECK(intEqual(result, currentPosition));
        }
    );

    ////

    COMPILE_ASSERT(sizeof(CpuAddrU) <= sizeof(size_t));
    auto result = ::read(currentHandle, dataPtr, size_t{dataSize});

    ////

    CharArray errorMsg = STR("Cannot read %0 bytes at offset %1 from file %2: %3");
    REQUIRE_TRACE4(intEqual(result, dataSize), errorMsg, dataSize, currentPosition, currentFilename, getLastError());

    ////

    restorePositionCleanup.cancel();

    currentPosition += dataSize;

    returnTrue;
}

//================================================================
//
// BinaryFileLinux::write
//
//================================================================

stdbool BinaryFileLinux::write(const void* dataPtr, CpuAddrU dataSize, stdPars(FileDiagKit))
{
    REQUIRE(currentHandle != -1);

    ////

    REQUIRE(dataSize <= typeMax<uint64>() - currentSize);

    ////

    REMEMBER_CLEANUP_EX
    (
        restorePositionCleanup,
        {
            auto result = lseek(currentHandle, currentPosition, SEEK_SET);
            DEBUG_BREAK_CHECK(intEqual(result, currentPosition));
        }
    );

    ////

    COMPILE_ASSERT(sizeof(CpuAddrU) <= sizeof(size_t));
    auto result = ::write(currentHandle, dataPtr, size_t{dataSize});

    CharArray errorMsg = STR("Cannot write %0 bytes at offset %1 to file %2: %3");
    REQUIRE_TRACE4(intEqual(result, dataSize), errorMsg, dataSize, currentPosition, currentFilename, getLastError());

    ////

    restorePositionCleanup.cancel();

    currentPosition += dataSize;
    currentSize = maxv(currentSize, currentPosition);

    returnTrue;
}

//----------------------------------------------------------------

#endif
