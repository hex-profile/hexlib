#if defined(_WIN32)

#include "connectionWin32.h"

////

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

////

#include <mutex>

#include "formattedOutput/requireMsg.h"
#include "win32/errorWin32.h"
#include "storage/rememberCleanup.h"
#include "errorLog/debugBreak.h"
#include "numbers/int/intType.h"

namespace connection {

using namespace std;

ConnectionWin32 sss; // ```

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Check types.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

COMPILE_ASSERT(TYPE_EQUAL(Socket, SOCKET));
COMPILE_ASSERT(invalidSocket == INVALID_SOCKET);

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

    WORD version = MAKEWORD(2, 2);
    WSADATA data;
    int err = WSAStartup(version, &data);
    REQUIRE_MSG1(err == 0, STR("Connection: Winsock init failed: %0"), ErrorWin32(err));
    REMEMBER_CLEANUP_EX(wsaCleanup, DEBUG_BREAK_CHECK(WSACleanup() == 0)); // WSAStartup returned 0, remember to cleanup.
    REQUIRE_MSG(data.wVersion == version, STR("Connection: Winsock: Cannot find version 2.2."));

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
// ConnectionWin32
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

stdbool ConnectionWin32::open(const Address& address, stdPars(DiagnosticKit))
{
    close();

    //----------------------------------------------------------------
    //
    // WinSock library.
    //
    //----------------------------------------------------------------

    require(winSockLib.open(stdPass));
    REMEMBER_CLEANUP_EX(winSockCleanup, winSockLib.close());

    //----------------------------------------------------------------
    //
    // Create a socket.
    //
    //----------------------------------------------------------------

    theSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    REQUIRE_MSG1(theSocket != invalidSocket, STR("Connection: Cannot create socket: %0"), ErrorWin32(WSAGetLastError()));
    REMEMBER_CLEANUP_EX(closeSocketCleanup, {DEBUG_BREAK_CHECK(closesocket(theSocket) == 0); theSocket = invalidSocket;});

    //----------------------------------------------------------------
    //
    // Port string.
    //
    //----------------------------------------------------------------

    char portStr[64];
    REQUIRE(sprintf(portStr, "%d", int(address.port)) >= 1);

    //----------------------------------------------------------------
    //
    // Get address info.
    //
    //----------------------------------------------------------------

    addrinfo hints;
    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* ai = nullptr;
    auto aiErr = getaddrinfo(address.host, portStr, &hints, &ai);
    REQUIRE_MSG3(aiErr == 0, STR("Connection: Get address info failed for %0:%1. %2"), 
        address.host, address.port, ErrorWin32(aiErr));

    REMEMBER_CLEANUP(freeaddrinfo(ai));

    //----------------------------------------------------------------
    //
    // Connect the socket.
    //
    //----------------------------------------------------------------

    REQUIRE_MSG3(connect(theSocket, ai->ai_addr, int(ai->ai_addrlen)) == 0, 
        STR("Connection: Cannot connect to %0:%1. %2"), address.host, address.port, ErrorWin32(WSAGetLastError()));

    ////

    theOpened = true;

    winSockCleanup.cancel();
    closeSocketCleanup.cancel();

    returnTrue;
}

//================================================================
//
// ConnectionWin32::close
//
//================================================================

void ConnectionWin32::close()
{
    if (theOpened)
    {
        // First close the socket, then the library.
        DEBUG_BREAK_CHECK(closesocket(theSocket) == 0);
        theSocket = invalidSocket;

        winSockLib.close();

        theOpened = false;
    }
}

//================================================================
//
// ConnectionWin32::send
//
//================================================================

stdbool ConnectionWin32::send(const void* dataPtr, size_t dataSize, stdPars(DiagnosticKit))
{
    REQUIRE(theOpened);

    ////

    COMPILE_ASSERT(INT_MAX <= SIZE_MAX);
    size_t maxRequestSize = INT_MAX;

    ////

    const char* currentPtr = (const char*) dataPtr;
    size_t currentSize = dataSize;

    ////

    while (currentSize > 0)
    {
        int requestSize = int(clampMax(currentSize, maxRequestSize));
        int actualSize = ::send(theSocket, currentPtr, requestSize, 0);

        REQUIRE_MSG1(actualSize >= 0, STR("Connection: Cannot send data. %0"), ErrorWin32(WSAGetLastError()));

        REQUIRE(actualSize <= requestSize);
        currentSize -= actualSize;
        currentPtr += actualSize;
    }

    returnTrue;
}

//================================================================
//
// ConnectionWin32::receive
//
//================================================================

stdbool ConnectionWin32::receive(void* dataPtr, size_t dataSize, size_t& actualDataSize, stdPars(DiagnosticKit))
{
    REQUIRE(theOpened);

    REQUIRE(dataSize <= size_t(INT_MAX));
    int actualSize = ::recv(theSocket, (char*) dataPtr, int(dataSize), 0);
    REQUIRE_MSG1(actualSize >= 0, STR("Connection: Cannot receive data. %1"), ErrorWin32(WSAGetLastError()));

    actualDataSize = size_t(actualSize);
    returnTrue;
}

//----------------------------------------------------------------

}

#endif
