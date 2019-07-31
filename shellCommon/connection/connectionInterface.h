#pragma once

#include "userOutput/diagnosticKit.h"
#include "numbers/int/intBase.h"
#include "stdFunc/stdFunc.h"

namespace connection {

//================================================================
//
// Host
//
//================================================================

using Host = const char*;

//================================================================
//
// Port
//
//================================================================

using Port = uint16;

//================================================================
//
// Address
//
//================================================================

struct Address
{
    Host host;
    Port port;
};

//================================================================
//
// Sending
//
//================================================================

struct Sending
{
    virtual stdbool send(const void* dataPtr, size_t dataSize, stdPars(DiagnosticKit)) =0;
};

//================================================================
//
// Receiving
//
//================================================================

struct Receiving
{
    virtual stdbool receive(void* dataPtr, size_t dataSize, size_t& actualDataSize, stdPars(DiagnosticKit)) =0;
};

//================================================================
//
// Opening
//
//================================================================

struct Opening
{
    virtual bool opened() const =0;
    virtual stdbool open(const Address& address, stdPars(DiagnosticKit)) =0;
    virtual void close() =0;
};

//================================================================
//
// Connection
//
//================================================================

struct Connection : public Opening, public Sending, public Receiving
{
};

//----------------------------------------------------------------

}
