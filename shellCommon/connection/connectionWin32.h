#pragma once

#include "connectionInterface.h"

namespace connection {

//================================================================
//
// Socket
//
//================================================================

using Socket = size_t;
constexpr Socket invalidSocket = Socket(-1);

//================================================================
//
// ConnectionWin32
//
//================================================================

class ConnectionWin32 : public Connection
{

public:

    ~ConnectionWin32();

public:

    virtual bool opened() const {return theStatus == Status::Opened;}
    virtual stdbool open(const Address& address, stdPars(Kit));
    virtual stdbool reopen(stdPars(Kit));
    virtual void close();

public:

    stdbool send(const void* dataPtr, size_t dataSize, stdPars(Kit));
    stdbool receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(Kit));

private:

    enum class Status {None, LibUsed, Opened};
    Status theStatus = Status::None;
    
    Socket theSocket = invalidSocket;
    void* addrInfo = nullptr;
};

//----------------------------------------------------------------

}
