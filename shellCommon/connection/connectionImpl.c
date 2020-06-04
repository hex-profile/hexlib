#include "connectionImpl.h"

////

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
#endif

#ifdef __linux__
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netdb.h>
#endif

////

#include <mutex>

#include "osErrors/errorWin32.h"
#include "osErrors/errorLinux.h"
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

#ifdef _WIN32

COMPILE_ASSERT(TYPE_EQUAL(Socket, SOCKET));
COMPILE_ASSERT(invalidSocket == INVALID_SOCKET);

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Common funcs.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// getSocketError
//
//================================================================

#if defined(_WIN32)

    inline auto getSocketError() {return ErrorWin32(WSAGetLastError());}

#elif defined(__linux__)

    inline auto getSocketError() {return ErrorLinux(errno);}

#else

    #error

#endif

//================================================================
//
// closesocket
//
//================================================================

#ifdef __linux__

inline int closesocket(int handle)
    {return close(handle);}

#endif

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

#ifdef _WIN32

    mutex lock;
    uint32 refCount = 0;

#endif

};

//================================================================
//
// WinSockLib::open
//
//================================================================

stdbool WinSockLib::open(stdPars(Kit))
{

#ifdef _WIN32

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

#endif

    returnTrue;
}

//================================================================
//
// WinSockLib::close
//
//================================================================

void WinSockLib::close()
{

#ifdef _WIN32

    lock_guard<mutex> guard(lock);

    if_not (refCount >= 1)
        return; // usage error

    --refCount;

    if (refCount == 0)
        DEBUG_BREAK_CHECK(WSACleanup() == 0);

#endif

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
// ConnectionImpl
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

State ConnectionImpl::state() const
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
// ConnectionImpl::reopen
//
//================================================================

stdbool ConnectionImpl::reopen(const Address& address, stdPars(Kit))
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
    memset(&hints, 0, sizeof(hints));

    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* ai = nullptr;

    REQUIRE_TRACE3(getaddrinfo(address.host, portStr, &hints, &ai) == 0,
        STR("Connection: Get address info failed for %0:%1. %2"), address.host, address.port, getSocketError());

    ////

    theAddrInfo = ai;
    theStatus = Status::Resolved;
    cleanInfo.cancel();

    returnTrue;
}

//================================================================
//
// ConnectionImpl::close
//
//================================================================

void ConnectionImpl::close()
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
// ConnectionImpl::reconnect
//
//================================================================

stdbool ConnectionImpl::reconnect(stdPars(Kit))
{
    REQUIRE(theStatus >= Status::Resolved);

    disconnect();

    //----------------------------------------------------------------
    //
    // Create a socket.
    //
    //----------------------------------------------------------------

    theSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    REQUIRE_TRACE1(theSocket != invalidSocket, STR("Connection: Cannot create socket: %0"), getSocketError());
    REMEMBER_CLEANUP_EX(closeSocketCleanup, {DEBUG_BREAK_CHECK(closesocket(theSocket) == 0); theSocket = invalidSocket;});

    //----------------------------------------------------------------
    //
    // Linger.
    //
    //----------------------------------------------------------------

    /*
    LINGER linger = {1, 60};
    REQUIRE(setsockopt(theSocket, SOL_SOCKET, SO_LINGER, (char*) &linger, sizeof(linger)) == 0);
    */

    //----------------------------------------------------------------
    //
    // Connect the socket.
    //
    //----------------------------------------------------------------

    auto ai = (addrinfo*) theAddrInfo;

    REQUIRE_TRACE3(connect(theSocket, ai->ai_addr, int(ai->ai_addrlen)) == 0, 
        STR("Connection: Cannot connect to %0:%1. %2"), theHost.cstr(), thePort, getSocketError());

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
// ConnectionImpl::disconnect
//
//================================================================

void ConnectionImpl::disconnect()
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
// ConnectionImpl::~ConnectionImpl
//
//================================================================

ConnectionImpl::~ConnectionImpl()
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
// ConnectionImpl::send
//
//================================================================

stdbool ConnectionImpl::send(const void* dataPtr, size_t dataSize, stdPars(Kit))
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

        REQUIRE_TRACE1(actualSize >= 0, STR("Connection: Cannot send data. %0"), getSocketError());

        REQUIRE(actualSize <= requestSize);
        currentSize -= size_t(actualSize);
        currentPtr += size_t(actualSize);
    }

    returnTrue;
}

//================================================================
//
// ConnectionImpl::receive
//
//================================================================

stdbool ConnectionImpl::receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(Kit))
{
    REQUIRE(theStatus == Status::Connected);

    REQUIRE(dataSize <= size_t(INT_MAX));
    int actualSize = ::recv(theSocket, (char*) dataPtr, int(dataSize), 0);
    REQUIRE_TRACE1(actualSize >= 0, STR("Connection: Cannot receive data. %0"), getSocketError());

    receivedSize = size_t(actualSize);
    returnTrue;
}

//----------------------------------------------------------------

}
