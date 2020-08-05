#pragma once

#include "configFile/cfgStringEnv.h"
#include "interfaces/fileToolsKit.h"
#include "stdFunc/stdFunc.h"
#include "errorLog/errorLogKit.h"

namespace cfgVarsImpl {

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
    // Read variables from a file, returns success flag.
    // If the file cannot be open, does not clear variables in memory.
    //

    virtual stdbool loadFromFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit)) =0;

    //
    // Save variables to a file, returns success flag.
    //

    virtual stdbool saveToFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit)) const =0;

};

//================================================================
//
// FileEnvSTL
//
// FileEnv core implementation
//
//================================================================

class FileEnvSTL : public FileEnv
{

public:

    virtual bool get(const NameContainer& name, String& value, String& valueComment, String& blockComment) const
        {return !impl ? false : impl->get(name, value, valueComment, blockComment);}

    virtual bool set(const NameContainer& name, const String& value, const String& valueComment, const String& blockComment)
        {return !impl ? false : impl->set(name, value, valueComment, blockComment);}

    void eraseAll()
        {if (impl) impl->eraseAll();}

    stdbool loadFromFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit))
    {
        require(impl);
        return impl->loadFromFile(filename, fileTools, stdPassThru);
    }

    stdbool saveToFile(const CharType* filename, FileTools& fileTools, stdPars(ErrorLogKit)) const
    {
        require(impl);
        return impl->saveToFile(filename, fileTools, stdPassThru);
    }

public:

    inline bool created() const {return impl != 0;}

public:

    FileEnvSTL();
    ~FileEnvSTL();

private:

    FileEnv* impl;

};

//----------------------------------------------------------------

}
