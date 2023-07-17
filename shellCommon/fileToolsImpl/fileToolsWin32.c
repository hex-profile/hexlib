#include "interfaces/fileTools.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <memory>

#include "storage/rememberCleanup.h"

namespace fileTools {

//================================================================
//
// isFileEx
//
//================================================================

template <typename Check>
inline bool isFileEx(const CharType* filename, const Check& check)
{
    //
    // only FindFirstFile - GetFileAttributes sees not all files
    //

    WIN32_FIND_DATA data;

    HANDLE handle = FindFirstFile(filename, &data);

    bool yes = (handle != INVALID_HANDLE_VALUE) && check(data);

    if (handle != INVALID_HANDLE_VALUE)
        FindClose(handle);

    return yes;
}

//================================================================
//
// isFile
//
//================================================================

bool isFile(const CharType* filename)
{
    auto check = [] (auto& data) {return (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;};

    return isFileEx(filename, check);
}

//================================================================
//
// isDir
//
//================================================================

bool isDir(const CharType* filename)
{
    auto check = [] (auto& data) {return (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;};

    return isFileEx(filename, check);
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

    ensure(result({pathPtr, n}));
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
