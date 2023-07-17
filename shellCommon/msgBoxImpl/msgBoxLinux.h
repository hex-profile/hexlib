#pragma once

#include "interfaces/msgBox.h"

//================================================================
//
// MsgBoxLinux
//
//================================================================

class MsgBoxLinux : public MsgBox
{

public:

    bool operator () (const CharType* message, MsgKind msgKind);

};
