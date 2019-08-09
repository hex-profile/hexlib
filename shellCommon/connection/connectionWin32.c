#if defined(_WIN32)

#include "connectionWin32.h"

////

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

////

#include <mutex>

#include "win32/errorWin32.h"
#include "storage/rememberCleanup.h"
#include "errorLog/debugBreak.h"
#include "numbers/int/intType.h"
#include "userOutput/errorLogEx.h"

namespace connection {

using namespace std;

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

    stdbool open(stdPars(Kit));
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

stdbool WinSockLib::open(stdPars(Kit))
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
    REQUIRE_TRACE1(err == 0, STR("Connection: Winsock init failed: %0"), ErrorWin32(err));
    REMEMBER_CLEANUP_EX(wsaCleanup, DEBUG_BREAK_CHECK(WSACleanup() == 0)); // WSAStartup returned 0, remember to cleanup.
    REQUIRE_TRACE(data.wVersion == version, STR("Connection: Winsock: Cannot find version 2.2."));

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

State ConnectionWin32::state() const
{
    State result = State::None;

    if (theStatus == Status::Resolved)
        result = State::Resolved;

    if (theStatus == Status::Connected)
        result = State::Connected;

    return result;
}

//================================================================
//
// ConnectionWin32::reopen
//
//================================================================

stdbool ConnectionWin32::reopen(const Address& address, stdPars(Kit))
{
    //----------------------------------------------------------------
    //
    // Close everything.
    //
    //----------------------------------------------------------------

    close();

    //----------------------------------------------------------------
    //
    // WinSock library.
    //
    //----------------------------------------------------------------

    if (theStatus == Status::None)
    {
        require(winSockLib.open(stdPass));
        theStatus = Status::LibUsed;
    }

    //----------------------------------------------------------------
    //
    // Remember host/port.
    //
    //----------------------------------------------------------------

    theHost = address.host;
    REQUIRE(def(theHost));
    thePort = address.port;

    REMEMBER_CLEANUP_EX(cleanInfo, {theHost.clear(); thePort = 0;});

    //----------------------------------------------------------------
    //
    // Port string.
    //
    //----------------------------------------------------------------

    REQUIRE(address.port >= 0 && address.port <= 0xFFFF); 
    const size_t maxDigits = 5; // at most 5 decimal digits.

    char portStr[maxDigits + 1];
    int n = sprintf(portStr, "%d", int(address.port));
    REQUIRE(n >= 1 && n <= maxDigits);

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

    REQUIRE_TRACE3(getaddrinfo(address.host, portStr, &hints, &ai) == 0,
        STR("Connection: Get address info failed for %0:%1. %2"), address.host, address.port, ErrorWin32(WSAGetLastError()));

    ////

    theAddrInfo = ai;
    theStatus = Status::Resolved;
    cleanInfo.cancel();

    returnTrue;
}

//================================================================
//
// ConnectionWin32::close
//
//================================================================

void ConnectionWin32::close()
{
    disconnect();

    ////

    if (theStatus == Status::Resolved)
    {
        freeaddrinfo((addrinfo*) theAddrInfo);
        theAddrInfo = nullptr;

        theHost.clear();
        thePort = 0;

        theStatus = Status::LibUsed; 
    }

    ////

    if (theStatus == Status::LibUsed)
    {
        // Keep the library. Free it only in the destructor.
    }
}

//================================================================
//
// ConnectionWin32::reconnect
//
//================================================================

stdbool ConnectionWin32::reconnect(stdPars(Kit))
{
    REQUIRE(theStatus >= Status::Resolved);

    disconnect();

    //----------------------------------------------------------------
    //
    // Create a socket.
    //
    //----------------------------------------------------------------

    theSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    REQUIRE_TRACE1(theSocket != invalidSocket, STR("Connection: Cannot create socket: %0"), ErrorWin32(WSAGetLastError()));
    REMEMBER_CLEANUP_EX(closeSocketCleanup, {DEBUG_BREAK_CHECK(closesocket(theSocket) == 0); theSocket = invalidSocket;});

    //----------------------------------------------------------------
    //
    // Connect the socket.
    //
    //----------------------------------------------------------------

    auto ai = (addrinfo*) theAddrInfo;

    REQUIRE_TRACE3(connect(theSocket, ai->ai_addr, int(ai->ai_addrlen)) == 0, 
        STR("Connection: Cannot connect to %0:%1. %2"), theHost.cstr(), thePort, ErrorWin32(WSAGetLastError()));

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    closeSocketCleanup.cancel();
    theStatus = Status::Connected;

    returnTrue;
}

//================================================================
//
// ConnectionWin32::disconnect
//
//================================================================

void ConnectionWin32::disconnect()
{
    if (theStatus == Status::Connected)
    {
        DEBUG_BREAK_CHECK(closesocket(theSocket) == 0);
        theSocket = invalidSocket;

        theStatus = Status::Resolved;
    }
}

//================================================================
//
// ConnectionWin32::~ConnectionWin32
//
//================================================================

ConnectionWin32::~ConnectionWin32()
{
    // Close everything.
    close();

    // Free the library reference.
    if (theStatus == Status::LibUsed)
    {
        winSockLib.close();
        theStatus = Status::None;
    }
}

//================================================================
//
// ConnectionWin32::send
//
//================================================================

stdbool ConnectionWin32::send(const void* dataPtr, size_t dataSize, stdPars(Kit))
{
    REQUIRE(theStatus == Status::Connected);

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

        REQUIRE_TRACE1(actualSize >= 0, STR("Connection: Cannot send data. %0"), ErrorWin32(WSAGetLastError()));

        REQUIRE(actualSize <= requestSize);
        currentSize -= size_t(actualSize);
        currentPtr += size_t(actualSize);
    }

    returnTrue;
}

//================================================================
//
// ConnectionWin32::receive
//
//================================================================

stdbool ConnectionWin32::receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(Kit))
{
    REQUIRE(theStatus == Status::Connected);

    REQUIRE(dataSize <= size_t(INT_MAX));
    int actualSize = ::recv(theSocket, (char*) dataPtr, int(dataSize), 0);
    REQUIRE_TRACE1(actualSize >= 0, STR("Connection: Cannot receive data. %0"), ErrorWin32(WSAGetLastError()));

    receivedSize = size_t(actualSize);
    returnTrue;
}

//----------------------------------------------------------------

}

#endif
