#pragma once

#include "charType/charArray.h"
#include "kit/kit.h"
#include "numbers/int/intBase.h"

//================================================================
//
// GetString
//
// Interface to return a string.
//
//================================================================

struct GetString
{
    virtual bool setBuffer(const CharType* bufArray, size_t bufSize) =0;
};

//================================================================
//
// FileTime
//
// Some type to compare using ==, >, <, >=, <=.
// There is no other meaning.
//
//================================================================

using FileTime = int64;

//================================================================
//
// FileSize
//
//================================================================

using FileSize = uint64;

//================================================================
//
// FileTools
//
//================================================================

struct FileTools
{
    // Check if file exists.
    virtual bool fileExists(const CharType* filename) =0;

    // Get file size.
    virtual bool getFileSize(const CharType* filename, FileSize& result) =0;

    // Returns the time of last file modification.
    virtual bool getChangeTime(const CharType* filename, FileTime& result) =0;

    // Delete file.
    virtual bool deleteFile(const CharType* filename) =0;

    // Atomic rename, replacing destination file.
    virtual bool renameFile(const CharType* oldName, const CharType* newName) =0;

    // Expand file name to full absolute path.
    virtual bool expandPath(const CharType* filename, GetString& result) =0;

    // Make directory
    virtual bool makeDirectory(const CharType* filename) =0;
};
