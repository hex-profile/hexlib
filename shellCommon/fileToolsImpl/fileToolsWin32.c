#ifdef _WIN32

#include "fileToolsWin32.h"

#include <windows.h>
#include <memory>

#include "storage/rememberCleanup.h"

//================================================================
//
// FileToolsWin32::deleteFile
// FileToolsWin32::renameFile
//
//================================================================

bool FileToolsWin32::deleteFile(const CharType* name)
    {return DeleteFile(name) != 0;}

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

    require(size >= 1);
    require(size <= 65536);

    CharType* pathPtr = new (std::nothrow) CharType[size];
    require(pathPtr);
    REMEMBER_CLEANUP1(delete[] pathPtr, CharType*, pathPtr);

    DWORD n = GetFullPathName(filename, size, pathPtr, &dummy);
    require(n == size - 1);

    require(result.setBuffer(pathPtr, n));
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
