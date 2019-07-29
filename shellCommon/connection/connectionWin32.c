#if defined(_WIN32)

#include "connectionWin32.h"

////

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

////

#include <mutex>

#include "formattedOutput/requireMsg.h"
#include "win32/errorWin32.h"
#include "storage/rememberCleanup.h"
#include "errorLog/debugBreak.h"

namespace connection {

using namespace std;

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// WSA library
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

class WinSockLib
{

public:

    stdbool open(stdPars(DiagnosticKit));
    void close();

private:

    mutex lock;
    uint32 refCount = 0;

};

//================================================================
//
// WinSockLib::open
//
//================================================================

stdbool WinSockLib::open(stdPars(DiagnosticKit))
{
    lock_guard<mutex> guard(lock);

    ////

    if (refCount != 0)
    {
        ++refCount;
        returnTrue;
    }

    ////

    WORD wVersionRequested = MAKEWORD(2, 2);
    WSADATA wsaData;
    int err = WSAStartup(wVersionRequested, &wsaData);
    REQUIRE_MSG1(err == 0, STR("Connection: Winsock: Init failed: %0"), ErrorWin32(err));
    REMEMBER_CLEANUP_EX(wsaCleanup, DEBUG_BREAK_CHECK(WSACleanup() == 0)); // WSAStartup returned 0, remember to cleanup.

    REQUIRE_MSG(LOBYTE(wsaData.wVersion) == 2 && HIBYTE(wsaData.wVersion) == 2,
        STR("Connection: Winsock: Cannot find version 2.2"));

    ////

    wsaCleanup.cancel();
    ++refCount;
    returnTrue;
}

//================================================================
//
// WinSockLib::close
//
//================================================================

void WinSockLib::close()
{
    lock_guard<mutex> guard(lock);

    if_not (refCount >= 1)
        return; // usage error

    --refCount;

    if (refCount == 0)
        DEBUG_BREAK_CHECK(WSACleanup() == 0);
}

//================================================================
//
// Global instance.
//
//================================================================

WinSockLib winSockLib;

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
//
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================


//----------------------------------------------------------------

}

#endif
