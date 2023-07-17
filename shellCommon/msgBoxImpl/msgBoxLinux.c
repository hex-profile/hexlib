#include "msgBoxLinux.h"

#include <stdio.h>

//================================================================
//
// MsgBoxLinux::operator ()
//
//================================================================

bool MsgBoxLinux::operator ()(const CharType* message, MsgKind msgKind)
{
    auto result = fprintf(msgKind == msgInfo ? stdout : stderr, "%s\n", message);
    return (result >= 0);
}
