#pragma once

#include "connectionInterface.h"
#include "simpleString/simpleString.h"
#include "storage/nonCopyable.h"

namespace connection {

//================================================================
//
// Socket
//
//================================================================

#if defined(_WIN32)

    using Socket = size_t;

#elif defined(__linux__)

    using Socket = int;

#else

    #error

#endif

////

constexpr Socket invalidSocket = Socket(-1);

//================================================================
//
// ConnectionImpl
//
//================================================================

class ConnectionImpl : public Connection, private NonCopyable
{

public:

    ~ConnectionImpl();

public:

    virtual State state() const;

    virtual stdbool reopen(const Address& address, stdPars(Kit));
    virtual void close();

    virtual stdbool reconnect(stdPars(Kit));
    virtual void disconnect();

public:

    stdbool send(const void* dataPtr, size_t dataSize, stdPars(Kit));
    stdbool receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(Kit));

private:

    enum class Status {None, LibUsed, Resolved, Connected};
    Status theStatus = Status::None;

    SimpleString theHost;
    Port thePort = 0;

    Socket theSocket = invalidSocket;
    void* theAddrInfo = nullptr;
};

//----------------------------------------------------------------

}
