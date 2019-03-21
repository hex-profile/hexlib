#pragma once

#include "interfaces/fileTools.h"

//================================================================
//
// FileToolsLinux
//
//================================================================

class FileToolsLinux : public FileTools
{

public:

    bool deleteFile(const CharType* name);
    bool renameFile(const CharType* oldName, const CharType* newName);
    bool expandPath(const CharType* filename, GetString& result);
    bool makeDirectory(const CharType* filename);

};
