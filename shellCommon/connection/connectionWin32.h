#pragma once

#include "connectionInterface.h"

namespace connection {

//================================================================
//
// Socket
//
//================================================================

using Socket = unsigned int;
constexpr Socket invalidSocket = Socket(-1);

//================================================================
//
// ConnectionWin32
//
//================================================================

class ConnectionWin32 : public Connection
{

public:

    ~ConnectionWin32() {close();}

public:

    virtual bool opened() const {return theOpened;}
    virtual stdbool open(const Address& address, stdPars(DiagnosticKit));
    virtual void close();

public:

    stdbool send(const void* dataPtr, size_t dataSize, stdPars(DiagnosticKit));
    stdbool receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(DiagnosticKit));

private:

    bool theOpened = false;
    Socket theSocket = invalidSocket;

};

inline void test111()
{
    ConnectionWin32 test;// ```
}

//----------------------------------------------------------------

}
