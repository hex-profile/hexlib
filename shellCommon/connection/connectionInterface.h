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

using Kit = KitCombine<DiagnosticKit, ProfilerKit>;

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

    inline bool isResolved() const
        {return state() >= State::Resolved;}

    inline bool isConnected() const
        {return state() >= State::Connected;}

    // Resolve address and remember it.
    // On success, the state becomes RESOLVED.
    // On failure, the state drops to NONE.
    virtual stdbool reopen(const Address& address, stdPars(Kit)) =0;
    // Close everything. The state becomes NONE.
    virtual void close() =0;

    // Re-connects to the resolved address. Requires RESOLVED+ state.
    // On success, the state becomes CONNECTED.
    // On failure, the state drops to RESOLVED.
    virtual stdbool reconnect(stdPars(Kit)) =0;
    // Disconnect. The state drops to RESOLVED (unless it was NONE).
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
