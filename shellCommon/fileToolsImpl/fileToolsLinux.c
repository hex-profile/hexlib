#include "interfaces/fileTools.h"

#ifdef __linux__

#include <memory>
#include <algorithm>

#include <sys/stat.h>
#include <linux/limits.h>
#include <unistd.h>

#include <string.h>

#include "storage/rememberCleanup.h"
#include "stlString/stlString.h"
#include "compileTools/errorHandling.h"

namespace fileTools {

//================================================================
//
// isFile
//
//================================================================

bool isFile(const CharType* filename)
{
    return access(filename, F_OK) != -1;
}

//================================================================
//
// isDir
//
//================================================================

bool isDir(const CharType* filename)
{
    struct stat sb;
    return stat(filename, &sb) == 0 && S_ISDIR(sb.st_mode);
}

//================================================================
//
// getFileSize
//
//================================================================

bool getFileSize(const CharType* filename, FileSize& result)
{
    struct stat tmp;
    ensure(stat(filename, &tmp) == 0);

    ensure(tmp.st_size >= 0);
    result = tmp.st_size;

    return true;
}

//================================================================
//
// getChangeTime
//
//================================================================

bool getChangeTime(const CharType* filename, FileTime& result)
{
    struct stat tmp;
    ensure(stat(filename, &tmp) == 0);

    result = tmp.st_mtime;
    return true;
}

//================================================================
//
// deleteFile
// renameFile
//
//================================================================

bool deleteFile(const CharType* filename)
{
    return remove(filename) == 0;
}

bool renameFile(const CharType* oldName, const CharType* newName)
{
    return rename(oldName, newName) == 0;
}

//================================================================
//
// makeDirectory
//
//================================================================

bool makeDirectory(const CharType* filename)
{
    return mkdir(filename, 0777) == 0;
}

//================================================================
//
// expandPath
//
//================================================================

bool expandPath(const CharType* filename, GetString& result)
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
            result(charArrayFromPtr(realPath));
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

        for (; ;)
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
        result(charArrayFromStl(str));

    }
    catch (const std::exception& e)
    {
        return false;
    }

    return true;
}

//----------------------------------------------------------------

}

#endif
