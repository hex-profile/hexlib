#include "interfaces/fileTools.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <memory>

#include "storage/rememberCleanup.h"

namespace fileTools {

//================================================================
//
// fileExists
//
//================================================================

bool fileExists(const CharType* filename)
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
// getChangeTime
//
//================================================================

bool getChangeTime(const CharType* filename, FileTime& result)
{
    struct _stati64 tmp;
    ensure(_stati64(filename, &tmp) == 0);

    result = tmp.st_mtime;
    return true;
}

//================================================================
//
// getFileSize
//
//================================================================

bool getFileSize(const CharType* filename, FileSize& result)
{
    struct _stati64 tmp;
    ensure(_stati64(filename, &tmp) == 0);

    ensure(tmp.st_size >= 0);
    result = tmp.st_size;

    return true;
}

//================================================================
//
// deleteFile
// renameFile
//
//================================================================

bool deleteFile(const CharType* filename)
    {return DeleteFile(filename) != 0;}

bool renameFile(const CharType* oldName, const CharType* newName)
    {return MoveFileEx(oldName, newName, MOVEFILE_REPLACE_EXISTING) != 0;}

//================================================================
//
// expandPath
//
//================================================================

bool expandPath(const CharType* filename, GetString& result)
{
    CharType* dummy(0);
    DWORD size = GetFullPathName(filename, 0, NULL, &dummy);

    ensure(size >= 1);
    ensure(size <= 65536);

    CharType* pathPtr = new (std::nothrow) CharType[size];
    ensure(pathPtr);
    REMEMBER_CLEANUP(delete[] pathPtr);

    DWORD n = GetFullPathName(filename, size, pathPtr, &dummy);
    ensure(n == size - 1);

    ensure(result.setBuffer(pathPtr, n));
    return true;
}

//================================================================
//
// makeDirectory
//
//================================================================

bool makeDirectory(const CharType* filename)
{
    return CreateDirectory(filename, NULL) != 0;
}

//----------------------------------------------------------------

}

#endif
