#include "interruptConsole.h"

#if defined(_WIN32)
    #include <windows.h>
#endif

#if defined(__linux__)
    #include <signal.h>
#endif

//================================================================
//
// interruptSignal
//
//================================================================

volatile bool interruptSignal = false;

//================================================================
//
// InterruptConsole::operator()
//
//================================================================

bool InterruptConsole::operator() ()
{
    return interruptSignal;
}

//================================================================
//
// Win32
//
//================================================================

#if defined(_WIN32)

BOOL WINAPI consoleCtrlHandler(DWORD dwCtrlType)
{
    interruptSignal = true;
    return true;
}

InterruptConsole::InterruptConsole()
{
    interruptSignal = false;
    SetConsoleCtrlHandler(consoleCtrlHandler, TRUE);
}

#endif

//================================================================
//
// Linux
//
//================================================================

#if defined(__linux__)

void signalHandler(int)
{
    interruptSignal = true;
}

InterruptConsole::InterruptConsole()
{
    struct sigaction handler;
    handler.sa_handler = signalHandler;
    sigemptyset(&handler.sa_mask);
    handler.sa_flags = 0;
    sigaction(SIGINT, &handler, NULL);
}

#endif
