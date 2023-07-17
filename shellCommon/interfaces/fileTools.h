#pragma once

#include "charType/charArray.h"
#include "kit/kit.h"
#include "numbers/int/intBase.h"
#include "storage/adapters/callable.h"

namespace fileTools {

//================================================================
//
// GetString
//
// Interface to return a string.
//
//================================================================

using GetString = Callable<bool (const CharArray& str)>;

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
// File tools.
//
//================================================================

// Check if file exists.
bool isFile(const CharType* filename);
bool isDir(const CharType* filename);

// Get file size.
bool getFileSize(const CharType* filename, FileSize& result);

// Returns the time of last file modification.
bool getChangeTime(const CharType* filename, FileTime& result);

// Delete file.
bool deleteFile(const CharType* filename);

// Atomic rename, replacing destination file.
bool renameFile(const CharType* oldName, const CharType* newName);

// Expand file name to full absolute path.
bool expandPath(const CharType* filename, GetString& result);

// Make directory
bool makeDirectory(const CharType* filename);

//----------------------------------------------------------------

}

using fileTools::FileTime;
using fileTools::FileSize;
