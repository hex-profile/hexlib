#include "setThreadName.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "podVector/podVector.h"
#include "compileTools/blockExceptionsSilent.h"

//================================================================
//
// setThreadName
//
//================================================================

bool setThreadName(const CharArray& name)
{
    boolFuncExceptBegin;

#if defined(_WIN32)

    ensure(name.size <= TYPE_MAX(size_t) - 1);

    PodVector<wchar_t> wideBuf;
    wideBuf.resize(name.size + 1, false);

    ////

    for_count (i, name.size)
        wideBuf[i] = name.ptr[i];

    wideBuf[name.size] = 0;

    ////

    ensure(SetThreadDescription(GetCurrentThread(), wideBuf.data()) != 0);

#endif

    boolFuncExceptEnd;
}
