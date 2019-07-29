#ifdef _WIN32

#include "fileToolsWin32.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <memory>

#include "storage/rememberCleanup.h"

//================================================================
//
// FileToolsWin32::fileExists
//
//================================================================

bool FileToolsWin32::fileExists(const CharType* filename)
{
    //
    // only FindFirstFile - GetFileAttributes sees not all files
    //

    WIN32_FIND_DATA tmp;
    HANDLE handle = FindFirstFile(filename, &tmp);
    bool yes = (handle != INVALID_HANDLE_VALUE);
    if (handle != INVALID_HANDLE_VALUE) FindClose(handle);
    return yes;
}

//================================================================
//
// FileToolsWin32::getChangeTime
//
//================================================================

bool FileToolsWin32::getChangeTime(const CharType* filename, FileTime& result)
{
    struct _stati64 tmp;
    ensure(_stati64(filename, &tmp) == 0);

    result = tmp.st_mtime;
    return true;
}

//================================================================
//
// FileToolsWin32::getFileSize
//
//================================================================

bool FileToolsWin32::getFileSize(const CharType* filename, FileSize& result)
{
    struct _stati64 tmp;
    ensure(_stati64(filename, &tmp) == 0);

    ensure(tmp.st_size >= 0);
    result = tmp.st_size;

    return true;
}

//================================================================
//
// FileToolsWin32::deleteFile
// FileToolsWin32::renameFile
//
//================================================================

bool FileToolsWin32::deleteFile(const CharType* filename)
    {return DeleteFile(filename) != 0;}

bool FileToolsWin32::renameFile(const CharType* oldName, const CharType* newName)
    {return MoveFileEx(oldName, newName, MOVEFILE_REPLACE_EXISTING) != 0;}

//================================================================
//
// FileToolsWin32::expandPath
//
//================================================================

bool FileToolsWin32::expandPath(const CharType* filename, GetString& result)
{
    CharType* dummy(0);
    DWORD size = GetFullPathName(filename, 0, NULL, &dummy);

    ensure(size >= 1);
    ensure(size <= 65536);

    CharType* pathPtr = new (std::nothrow) CharType[size];
    ensure(pathPtr);
    REMEMBER_CLEANUP1(delete[] pathPtr, CharType*, pathPtr);

    DWORD n = GetFullPathName(filename, size, pathPtr, &dummy);
    ensure(n == size - 1);

    ensure(result.setBuffer(pathPtr, n));
    return true;
}

//================================================================
//
// FileToolsWin32::makeDirectory
//
//================================================================

bool FileToolsWin32::makeDirectory(const CharType* filename)
{
    return CreateDirectory(filename, NULL) != 0;
}

//----------------------------------------------------------------

#endif
