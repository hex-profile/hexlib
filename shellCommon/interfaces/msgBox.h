#pragma once

#include "userOutput/msgLog.h"
#include "charType/charType.h"

//================================================================
//
// MsgBox
//
// Should support newline character inside the string.
//
//================================================================

struct MsgBox
{
    virtual bool operator () (const CharType* message, MsgKind msgKind) =0;
};
