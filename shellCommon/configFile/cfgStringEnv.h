#pragma once

#include <string>
#include <deque>

#include "cfg/cfgInterfaceFwd.h"
#include "charType/charType.h"

namespace cfgVarsImpl {

//================================================================
//
// String
//
//================================================================

using String = std::basic_string<CharType>;

//================================================================
//
// NamePart
//
//================================================================

struct NamePart
{
    String desc;

    inline NamePart()
        {}

    inline NamePart(const String& desc)
        : desc(desc) {}
};

//================================================================
//
// NameContainer
//
//================================================================

using NameContainer = std::deque<NamePart>;

//================================================================
//
// StringEnv
//
//----------------------------------------------------------------
//
// String environment interface.
//
// Allows to read/write/comment string cfg vars.
//
// Functions do not throw exceptions!
//
//================================================================

struct StringEnv
{

    //
    // Get variable.
    // The returned pointer will be valid until any modification of StringEnv.
    //

    virtual bool get(const NameContainer& name, String& value, String& valueComment, String& blockComment) const =0;

    //
    // Set variable and comment
    //

    virtual bool set(const NameContainer& name, const String& value, const String& valueComment, const String& blockComment) =0;

    //
    // Erase all variables from memory.
    //

    virtual void eraseAll() =0;

};

//----------------------------------------------------------------

}
