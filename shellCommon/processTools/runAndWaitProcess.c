#include "runAndWaitProcess.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#if defined(__linux__)
    #include <cstdlib>
#endif

#include "osErrors/errorWin32.h"
#include "osErrors/errorLinux.h"
#include "userOutput/printMsg.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// runAndWaitProcess (WIN32)
//
//================================================================

#if defined(_WIN32)

void runAndWaitProcess(const CharType* cmdLine, stdPars(MsgLogKit))
{
    STARTUPINFO si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    if_not (CreateProcess(NULL, (CharType*) cmdLine, NULL, NULL, false, 0, NULL, NULL, &si, &pi) != 0)
    {
        printMsg(kit.msgLog, STR("Cannot launch `%`, `%`"), cmdLine, ErrorWin32(GetLastError()), msgErr);
        kit.msgLog.update();
        returnFalse;
    }

    REMEMBER_CLEANUP(CloseHandle(pi.hProcess));
    REMEMBER_CLEANUP(CloseHandle(pi.hThread));

    // Wait process
    WaitForSingleObject(pi.hProcess, INFINITE);
}

//================================================================
//
// runAndWaitProcess (Linux)
//
//================================================================

#elif defined(__linux__)

void runAndWaitProcess(const CharType* cmdLine, stdPars(MsgLogKit))
{
    printMsg(kit.msgLog, STR("Editing: %"), cmdLine);
    kit.msgLog.update();

    auto result = system(cmdLine);

    if (result != 0)
    {
        printMsg(kit.msgLog, STR("Cannot launch `%`: %."), cmdLine, ErrorLinux(result), msgErr);
        kit.msgLog.update();
        returnFalse;
    }
}

//================================================================
//
// runAndWaitProcess (Not implemented)
//
//================================================================

#elif defined(__linux__)

void runAndWaitProcess(const CharType* cmdLine, stdPars(MsgLogKit))
{
    printMsg(kit.msgLog, STR("Cannot launch `%`: Not implemented."), cmdLine, msgErr);
    returnFalse;
}

#endif
