#include "runProcess.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "userOutput/printMsg.h"
#include "errorLog/errorLog.h"

//================================================================
//
// runProcess (Linux)
//
//================================================================

#if defined(__linux__)

void runProcess(const StlString& cmdLine, stdPars(RunProcessKit))
{
    // printMsg(kit.msgLog, STR("Launching %0"), cmdLine, msgInfo);

    int status = system(cmdLine.c_str());

    if_not (status == 0)
    {
        printMsg(kit.msgLog, STR("Cannot launch %0"), cmdLine, msgErr);
        returnFalse;
    }
}

#endif

//================================================================
//
// runProcess (Windows)
//
//================================================================

#if defined(_WIN32)

void runProcess(const StlString& cmdLine, stdPars(RunProcessKit))
{
    // printMsg(kit.msgLog, STR("Launching %0"), cmdLine, msgInfo);

    STARTUPINFO si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    bool createOk = CreateProcess(NULL, const_cast<char*>(cmdLine.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi) != 0;

    if_not (createOk)
    {
        printMsg(kit.msgLog, STR("Cannot launch %0"), cmdLine, msgErr);
        returnFalse;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode = 0;
    REQUIRE(GetExitCodeProcess(pi.hProcess, &exitCode) != 0);
    require(exitCode == 0); // success?

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}

#endif
