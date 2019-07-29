#include "logBufferDebugOutput.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "stlString/stlString.h"

//================================================================
//
// LogBufferDebugOutputThunk::add
//
//================================================================

bool LogBufferDebugOutputThunk::add(const CharArray& text, MsgKind kind, const TimeMoment& moment)
{

    try
    {

    #if defined(_WIN32)

        if (enabled)
        {
            StlString s(text.ptr, text.size);
            s.append(1, '\n');
            OutputDebugString(s.c_str());
        }

    #endif

    }
    catch (const std::exception&)
    {
    }

    return baseBuffer.add(text, kind, moment);

}

