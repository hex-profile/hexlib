#include "interruptConsole.h"

#include <atomic>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#if defined(__linux__)
    #include <signal.h>
#endif

//================================================================
//
// interruptSignal
// interruptSignaller
//
//================================================================

std::atomic<bool> interruptSignal{false};
std::atomic<InterruptSignaller*> interruptSignaller{nullptr};

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
// InterruptConsole::setSignaller
//
//================================================================

void InterruptConsole::setSignaller(InterruptSignaller* signaller)
{
    interruptSignaller = signaller;
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

    auto signaller = interruptSignaller.load();

    if (signaller)
        (*signaller)();

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

    auto signaller = interruptSignaller.load();

    if (signaller)
        (*signaller)();
}

InterruptConsole::InterruptConsole()
{
    struct sigaction handler;
    handler.sa_handler = signalHandler;
    sigemptyset(&handler.sa_mask);
    handler.sa_flags = 0;
    sigaction(SIGINT, &handler, NULL);
    sigaction(SIGTERM, &handler, NULL);
}

#endif
