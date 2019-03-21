#pragma once

#include "charType/charArray.h"
#include "kit/kit.h"

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
// FileTools
//
//================================================================

struct FileTools
{
    // Delete file.
    virtual bool deleteFile(const CharType* name) =0;

    // Atomic rename, replacing destination file.
    virtual bool renameFile(const CharType* oldName, const CharType* newName) =0;

    // Expand file name to full absolute path.
    virtual bool expandPath(const CharType* filename, GetString& result) =0;

    // Make directory
    virtual bool makeDirectory(const CharType* filename) =0;
};

//================================================================
//
// FileToolsKit
//
//================================================================

KIT_CREATE1(FileToolsKit, FileTools&, fileTools);
