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
// Standard parameters with trace support and some kit.
//
// stdPass
//
// Pass standard parameters, entering trace scope.
//
// stdPassKit
//
// The same, but with an explicitly specified kit.
//
// stdPassThru
// stdPassThruKit
//
// Pass standard parameters, without entering trace scope.
//
//================================================================

#define stdPars(Kit) \
    const Kit& kit, \
    TRACE_PARAMS(stdTraceName)

//----------------------------------------------------------------

#define stdPass \
    kit, \
    TRACE_PASS(stdTraceName, TRACE_AUTO_LOCATION)

#define stdPassKit(kit) \
    kit, \
    TRACE_PASS(stdTraceName, TRACE_AUTO_LOCATION)

//----------------------------------------------------------------

#define stdPassThru \
    kit, \
    TRACE_PASSTHRU(stdTraceName)

#define stdPassThruKit(kit) \
    kit, \
    TRACE_PASSTHRU(stdTraceName)

//----------------------------------------------------------------

#define stdPassLocMsg(msg) \
    kit, \
    TRACE_PASS(stdTraceName, TRACE_AUTO_LOCATION_MSG(msg))

//----------------------------------------------------------------

#define stdPassKitLocation(kit, location) \
    kit, \
    TRACE_PASS(stdTraceName, location)

//----------------------------------------------------------------

#define stdParsMember(Kit) \
    Kit kit; \
    TRACE_MEMBER(stdTraceName);

#define stdParsCapture \
    kit(kit), \
    TRACE_CAPTURE(stdTraceName)

//================================================================
//
// Standard begin/end macros with trace stack.
//
//================================================================

#define stdBeginEx(elemCount, profName) \
    { \
        TRACE_REASSEMBLE(stdTraceName); \
        PROFILER_SCOPE_EX(TRACE_SCOPE(stdTraceName).location, elemCount, profName)

#define stdBegin \
    stdBeginEx(0, 0)

#define stdBeginElem(elemCount) \
    stdBeginEx(elemCount, 0)

#define stdBeginProfName(profActive, profName) \
    stdBeginEx(0, (profActive) ? (profName) : 0)

//----------------------------------------------------------------

#define stdEnterEx(newLocation, elemCount, profName) \
    TRACE_ENTER(stdTraceName, newLocation); \
    PROFILER_SCOPE_EX(TRACE_SCOPE(stdTraceName).location, elemCount, profName)

#define stdEnter \
    stdEnterEx(TRACE_AUTO_LOCATION, 0, 0)

#define stdEnterLoc(newLocation) \
    stdEnterEx(newLocation, 0, 0)

#define stdEnterProfName(profActive, profName) \
    stdEnterEx(TRACE_AUTO_LOCATION, 0, (profActive) ? (profName) : 0)

//================================================================
//
// Standard return from function -- just for style.
//
//================================================================

#define stdEnd \
        return true; \
    }

#define stdEndv \
        return; \
    }

#define stdEndEx(value) \
        return (value); \
    }

//================================================================
//
// NullKit
//
//================================================================

struct NullKit
{
    template <typename AnyType>
    sysinline NullKit(const AnyType&) {}
};

//================================================================
//
// stdNullPars
// stdPass
// stdPassThru
//
// Standard trace support, no kit.
//
//================================================================

#define stdNullPars \
    const NullKit&, \
    TRACE_PARAMS(stdTraceName)

#define stdNullPass \
    stdPassKit(0)

#define stdNullPassThru \
    stdPassThruKit(0)

#define stdNullBegin \
    { \
        TRACE_REASSEMBLE(stdTraceName);
