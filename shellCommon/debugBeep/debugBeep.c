#include "debugBeep.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

//================================================================
//
// debugBeep
//
//================================================================

void debugBeep()
{

#if defined(_WIN32)
    Beep(10000, 100);
#endif

}
