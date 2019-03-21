#pragma once

#include "interfaces/msgBox.h"

//================================================================
//
// MsgBoxWin32
//
//================================================================

class MsgBoxWin32 : public MsgBox
{

public:

    bool operator () (const CharType* message, MsgKind msgKind);

};
