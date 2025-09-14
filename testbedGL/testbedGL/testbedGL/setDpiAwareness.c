#include "setDpiAwareness.h"

#if defined(_WIN32)
    #include <windows.h>
#endif

#include "osErrors/errorWin32.h"
#include "userOutput/printMsgTrace.h"

//================================================================
//
// setDpiAwareness
//
//================================================================

void setDpiAwareness(stdPars(DiagnosticKit))
{

#if defined(_WIN32)

    REQUIRE_TRACE1(SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) != 0,
        STR("Setting DPI Awareness failed: %0"), ErrorWin32(GetLastError()));

#endif
}
