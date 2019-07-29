#pragma once

#include "connectionInterface.h"

namespace connection {

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

    virtual bool opened() const {return isOpened;}
    virtual stdbool open(const Address& address, float32 timeoutInSec, stdPars(DiagnosticKit));
    virtual void close();

public:

    virtual stdbool send(const void* dataPtr, size_t dataSize, float32 timeoutInSec, stdPars(DiagnosticKit));
    virtual stdbool receive(void* dataPtr, size_t dataSize, float32 timeoutInSec, size_t& actualDataSize, stdPars(DiagnosticKit));

private:

    bool isOpened = false;

};

//----------------------------------------------------------------

}
