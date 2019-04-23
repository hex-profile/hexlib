#ifdef __linux__

#include "fileToolsLinux.h"

#include <memory>
#include <algorithm>

#include <sys/stat.h>
#include <linux/limits.h>
#include <unistd.h>

#include <string.h>

#include "storage/rememberCleanup.h"
#include "stlString/stlString.h"

//================================================================
//
// FileToolsLinux::fileExists
//
//================================================================

bool FileToolsLinux::fileExists(const CharType* filename)
{
    return access(filename, F_OK) != -1;
}

//================================================================
//
// FileToolsLinux::getFileSize
//
//================================================================

bool FileToolsLinux::getFileSize(const CharType* filename, FileSize& result)
{
    struct stat tmp;
    require(stat(filename, &tmp) == 0);

    require(tmp.st_size >= 0);
    result = tmp.st_size;

    return true;
}

//================================================================
//
// FileToolsLinux::getChangeTime
//
//================================================================

bool FileToolsLinux::getChangeTime(const CharType* filename, FileTime& result)
{
    struct stat tmp;
    require(stat(filename, &tmp) == 0);

    result = tmp.st_mtime;
    return true;
}

//================================================================
//
// FileToolsLinux::deleteFile
// FileToolsLinux::renameFile
//
//================================================================

bool FileToolsLinux::deleteFile(const CharType* filename)
{
    return remove(filename) == 0;
}

bool FileToolsLinux::renameFile(const CharType* oldName, const CharType* newName)
{
    return rename(oldName, newName) == 0;
}

//================================================================
//
// FileToolsLinux::makeDirectory
//
//================================================================

bool FileToolsLinux::makeDirectory(const CharType* filename)
{
    return mkdir(filename, 0777) == 0;
}

//================================================================
//
// FileToolsLinux::expandPath
//
//================================================================

bool FileToolsLinux::expandPath(const CharType* filename, GetString& result)
{
    using namespace std;

    try
    {
        CharType actualpath[PATH_MAX+1];

        //----------------------------------------------------------------
        //
        // try to handle instantly
        //
        //----------------------------------------------------------------

        CharType* realPath = realpath(filename, actualpath);

        if (realPath)
        {
            result.setBuffer(realPath, strlen(realPath));
            return true;
        }

        //----------------------------------------------------------------
        //
        // if the path is not absolute, consider it to be based on the current directory
        //
        //----------------------------------------------------------------

        size_t filenameLength = strlen(filename);

        if (!filenameLength)
            return false;

        StlString prefixedFilename;

        if (filename[0] != '/')
        {
            prefixedFilename = "./";
            prefixedFilename += filename;
            filename = prefixedFilename.c_str();
            filenameLength = prefixedFilename.size();
        }

        //----------------------------------------------------------------
        //
        // cut dirs one-by-one from the end until realpath starts to work
        //
        //----------------------------------------------------------------

        const CharType* strBegin = filename;
        const CharType* strEnd = filename + filenameLength;
        const CharType* strCurrentEnd = strEnd;

        const CharType* slashBegin = "/";
        const CharType* slashEnd = slashBegin + 1;

        ////

        for (;;)
        {
            auto p = find_end(strBegin, strCurrentEnd, slashBegin, slashEnd);

            if (p == strCurrentEnd)
                break;

            strCurrentEnd = p;

            StlString mainPart(strBegin, strCurrentEnd);
            realPath = realpath(mainPart.c_str(), actualpath);

            if (realPath)
                break;
        }

        ////

        if (!realPath)
            return false;

        ////

        StlString str(realPath);
        str.append(strCurrentEnd, strEnd);
        result.setBuffer(str.data(), str.size());

    }
    catch (const std::exception& e)
    {
        return false;
    }

    return true;
}

//----------------------------------------------------------------

#endif
