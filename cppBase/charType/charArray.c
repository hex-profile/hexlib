#include "charArray.h"

#include <string.h>

//================================================================
//
// charArrayFromPtr
//
//================================================================

CharArray charArrayFromPtr(const CharType* cstring)
{
    return CharArray(cstring, CHARTYPE_SELECT(strlen, wcslen)(cstring));
}
