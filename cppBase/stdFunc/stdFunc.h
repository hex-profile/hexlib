#pragma once

#include "stdFunc/traceCallstack.h"
#include "stdFunc/profiler.h"

//================================================================
//
// Standard function macros:
//
// * Standard parameters for dragging interface "kits" and trace callstack.
//
// * Standard function begin and end macros, implementing
// trace callstack support and other tools (profiler).
//
//================================================================

//================================================================
//
// stdTraceName
//
// The name of standard trace parameter.
//
//================================================================

#define stdTraceName __trace

//================================================================
//
// stdPars
//
// Declare standard parameters with trace support and some kit.
//
//================================================================

#define stdPars(Kit) \
    const Kit& kit, \
    TRACE_PARAMS(stdTraceName)

//================================================================
//
// stdPass
//
// Pass standard parameters, entering scope.
//
// stdPassKit
//
// The same, but with an explicitly specified kit.
//
// stdPassThru
// stdPassThruKit
//
// Pass standard parameters, without entering scope.
//
//================================================================

#define stdPassEx(kit, location) \
    (PROFILER_FRAME_TEMPORARY(kit, location), kit), \
    TRACE_PASS(stdTraceName, location)

//----------------------------------------------------------------

#define stdPass \
    stdPassEx(kit, TRACE_AUTO_LOCATION)

#define stdPassKit(kit) \
    stdPassEx(kit, TRACE_AUTO_LOCATION)

#define stdPassLocationMsg(msg) \
    stdPassEx(kit, TRACE_AUTO_LOCATION_MSG(msg))

//----------------------------------------------------------------

#define stdPassThru \
    kit, \
    TRACE_PASSTHRU(stdTraceName)

#define stdPassThruKit(kit) \
    kit, \
    TRACE_PASSTHRU(stdTraceName)

//================================================================
//
// stdBegin*
//
// Standard beginning of a function.
//
//================================================================

#define stdBegin 

//================================================================
//
// stdEnter*
//
// Open a new scope inside a function.
//
//================================================================

#define stdEnterEx(newLocation, elemCount, profName) \
    TRACE_ENTER(stdTraceName, newLocation); \
    PROFILER_FRAME_ENTER_EX(kit, TRACE_SCOPE(stdTraceName).location, elemCount, profName)

#define stdEnter \
    TRACE_ENTER(stdTraceName, TRACE_AUTO_LOCATION); \
    PROFILER_FRAME_ENTER(kit, TRACE_SCOPE(stdTraceName).location)

//----------------------------------------------------------------

#define stdEnterLocation(newLocation) \
    stdEnterEx(newLocation, 0, 0)

#define stdEnterElemCount(elemCount) \
    stdEnterEx(TRACE_AUTO_LOCATION, elemCount, 0)

//================================================================
//
// stdEnd*
//
// Standard return from function -- just for style.
//
//================================================================

#define stdEnd \
    return true \

#define stdEndv \
    return \

#define stdEndEx(value) \
    return (value) \

//================================================================
//
// Scoped variants, just for convenience.
//
//================================================================

#define stdBeginScoped \
    stdBegin; {

#define stdEndScoped \
    stdEnd; }

#define stdEndExScoped(value) \
    stdEndEx(value); }

//================================================================
//
// NullKit
//
//================================================================

struct NullKit
{
    sysinline NullKit() {}

    template <typename AnyType>
    sysinline NullKit(const AnyType&) {}
};

extern const NullKit nullKit;

//================================================================
//
// stdNullPass
// stdNullPassThru
//
// Standard trace support, no kit.
// 
//================================================================

#define stdNullPars \
    const NullKit&, \
    TRACE_PARAMS(stdTraceName)

#define stdNullPass \
    stdPassKit(nullKit)

#define stdNullPassThru \
    stdPassThruKit(nullKit)

#define stdNullBegin \
    TRACE_REASSEMBLE(stdTraceName);

//================================================================
//
// Preparing for the remake of all hexlib projects 
// to exception-based error handling.
//
// Temporary types to find all error recovering places in code.
//
//================================================================

using stdvoid = void;
using stvoid = void;
// ```

////

class stdbool // [[nodiscard]]
{

public:
    
    stdbool(bool value)
        : value(value) {}

    operator bool () const
        {return value;}

private:

    bool value;

};

//================================================================
//
// stdDiscard
//
//================================================================

inline void suppressHelper(stdbool value) {}

#define stdDiscard(value) \
    suppressHelper(value)
