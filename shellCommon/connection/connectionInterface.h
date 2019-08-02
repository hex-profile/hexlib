#pragma once

#include "userOutput/diagnosticKit.h"
#include "numbers/int/intBase.h"
#include "stdFunc/stdFunc.h"

namespace connection {

//================================================================
//
// Kit
//
//================================================================

KIT_COMBINE2(Kit, DiagnosticKit, ProfilerKit);

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
    virtual stdbool send(const void* dataPtr, size_t dataSize, stdPars(Kit)) =0;
};

//================================================================
//
// Receiving
//
//================================================================

struct Receiving
{
    virtual stdbool receive(void* dataPtr, size_t dataSize, size_t& receivedSize, stdPars(Kit)) =0;
};

//================================================================
//
// State
//
//================================================================

enum class State {None, Resolved, Connected};

//================================================================
//
// Opening
//
//================================================================

struct Opening
{
    // Get the current state.
    virtual State state() const =0;

    // Resolve address and remember it. On failure, the state drops to State::None.
    // Close everything.
    virtual stdbool reopen(const Address& address, stdPars(Kit)) =0;
    virtual void close() =0;

    // Connect to the resolved address. On failure, the state drops to State::Resolved.
    // Disconnect.
    virtual stdbool reconnect(stdPars(Kit)) =0;
    virtual void disconnect() =0;
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
