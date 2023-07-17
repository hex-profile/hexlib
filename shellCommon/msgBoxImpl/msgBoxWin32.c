#if defined(_WIN32)

#include "msgBoxWin32.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

//================================================================
//
// MsgBoxWin32::operator ()
//
//================================================================

bool MsgBoxWin32::operator ()(const CharType* message, MsgKind msgKind)
{
    int result = MessageBox
    (
        NULL,
        message,
        CT("Message"),

        msgKind == msgErr ? MB_ICONERROR :
        msgKind == msgWarn ? MB_ICONWARNING :
        MB_ICONINFORMATION
    );

    return result != 0;
}

//----------------------------------------------------------------

#endif
