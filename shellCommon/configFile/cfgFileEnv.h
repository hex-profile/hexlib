#pragma once

#include "configFile/cfgStringEnv.h"
#include "configFile/stringReceiver.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "storage/smartPtr.h"
#include "userOutput/errorLogExKit.h"

namespace cfgVarsImpl {

//================================================================
//
// FileEnvKit
//
//================================================================

using FileEnvKit = KitCombine<ErrorLogKit, ErrorLogExKit>;

//================================================================
//
// FileEnv interface
//
//----------------------------------------------------------------
//
// Keeps string text cfg variables and saves/loads them to/from a file.
//
//================================================================

struct FileEnv : public StringEnv
{

    //
    // If the file cannot be open, does not clear variables in memory.
    //

    virtual stdbool loadFromFile(const CharType* filename, stdPars(FileEnvKit)) =0;

    //
    // Save variables to a file.
    //

    virtual stdbool saveToFile(const CharType* filename, stdPars(FileEnvKit)) const =0;

    //
    // Load variables from a string.
    //

    virtual stdbool loadFromString(const CharArray& str, stdPars(FileEnvKit)) =0;

    //
    // Save variables to a string.
    //

    virtual stdbool saveToString(StringReceiver& receiver, stdPars(FileEnvKit)) const =0;

};

//================================================================
//
// FileEnvSTL
//
// FileEnv core implementation
//
//================================================================

struct FileEnvSTL : public FileEnv
{
    static UniquePtr<FileEnvSTL> create();
    virtual ~FileEnvSTL() {}
};

//----------------------------------------------------------------

}
