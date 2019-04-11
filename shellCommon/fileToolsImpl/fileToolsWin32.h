#pragma once

#include "interfaces/fileTools.h"

//================================================================
//
// FileToolsWin32
//
//================================================================

class FileToolsWin32 : public FileTools
{

public:

    bool fileExists(const CharType* filename);
    bool getFileSize(const CharType* filename, FileSize& result);
    bool getChangeTime(const CharType* filename, FileTime& result);
    bool deleteFile(const CharType* filename);
    bool renameFile(const CharType* oldName, const CharType* newName);
    bool expandPath(const CharType* filename, GetString& result);
    bool makeDirectory(const CharType* filename);

};
