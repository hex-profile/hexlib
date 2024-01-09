#pragma once

#include "stdFunc/traceCallstack.h"
#include "stdFunc/profiler.h"
#include "compileTools/errorHandling.h"

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
// TRACE_* common macros
//
//================================================================

#define stdTraceRoot \
    TRACE_ROOT(trace, TRACE_AUTO_LOCATION); \
    ERROR_HANDLING_ROOT

//================================================================
//
// stdPars
//
// Declare standard parameters with trace support and some kit.
//
//================================================================

#define stdPars(Kit) \
    const Kit& kit, \
    ERROR_HANDLING_PARAMS \
    TRACE_PARAMS(trace)

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
    ERROR_HANDLING_PASS \
    TRACE_PASS(trace, location) \
    ERROR_HANDLING_CHECK

//----------------------------------------------------------------

#define stdPass \
    stdPassEx(kit, TRACE_AUTO_LOCATION)

#define stdPassKit(kit) \
    stdPassEx(kit, TRACE_AUTO_LOCATION)

#define stdPassLocationMsg(msg) \
    stdPassEx(kit, TRACE_AUTO_LOCATION_MSG(msg))

//----------------------------------------------------------------

#define stdPassNoProfiling \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASS(trace, TRACE_AUTO_LOCATION) \
    ERROR_HANDLING_CHECK

//----------------------------------------------------------------

#define stdPassThru \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASSTHRU(trace) \
    ERROR_HANDLING_CHECK

#define stdPassThruKit(kit) \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASSTHRU(trace) \
    ERROR_HANDLING_CHECK

//================================================================
//
// stdPassNc*
//
//================================================================

#define stdPassExNc(kit, location) \
    (PROFILER_FRAME_TEMPORARY(kit, location), kit), \
    ERROR_HANDLING_PASS \
    TRACE_PASS(trace, location)

//----------------------------------------------------------------

#define stdPassNc \
    stdPassExNc(kit, TRACE_AUTO_LOCATION)

#define stdPassKitNc(kit) \
    stdPassExNc(kit, TRACE_AUTO_LOCATION)

#define stdPassLocationMsgNc(msg) \
    stdPassExNc(kit, TRACE_AUTO_LOCATION_MSG(msg))

//----------------------------------------------------------------

#define stdPassNoProfilingNc \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASS(trace, TRACE_AUTO_LOCATION)

//----------------------------------------------------------------

#define stdPassThruNc \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASSTHRU(trace)

#define stdPassThruKitNc(kit) \
    kit, \
    ERROR_HANDLING_PASS \
    TRACE_PASSTHRU(trace)

//================================================================
//
// stdEnter*
//
// Open a new scope inside a function.
//
//================================================================

#define stdEnterEx(newLocation, elemCount, profName) \
    TRACE_ENTER(trace, newLocation); \
    PROFILER_FRAME_ENTER_EX(kit, trace.location, elemCount, profName)

#define stdEnter \
    TRACE_ENTER(trace, TRACE_AUTO_LOCATION); \
    PROFILER_FRAME_ENTER(kit, trace.location)

//----------------------------------------------------------------

#define stdEnterLocation(newLocation) \
    stdEnterEx(newLocation, 0, 0)

#define stdEnterMsg(msg) \
    stdEnterEx(TRACE_AUTO_LOCATION_MSG(msg), 0, 0)

#define stdEnterElemCount(elemCount) \
    stdEnterEx(TRACE_AUTO_LOCATION, elemCount, 0)

//================================================================
//
// Scoped variants, just for convenience.
//
//================================================================

#define stdScopedBegin \
    {

#define stdScopedEnd \
    returnTrue; }

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
// stdParsNull
// stdPassNull
// stdPassNullThru
//
// Standard trace support, no kit.
//
//================================================================

#define stdParsNull \
    const NullKit&, \
    ERROR_HANDLING_PARAMS \
    TRACE_PARAMS(trace)

////

#define stdPassNull \
    stdPassKit(nullKit)

#define stdPassNullThru \
    stdPassThruKit(nullKit)

////

#define stdPassNullNc \
    stdPassKitNc(nullKit)

#define stdPassNullThruNc \
    stdPassThruKitNc(nullKit)

//================================================================
//
// stdParsMember
// stdParsCapture
//
//================================================================

#define stdParsMember(Kit) \
    const Kit& kit; \
    ERROR_HANDLING_MEMBER \
    TRACE_PARAMS(trace)

#define stdParsCapture \
    kit{kit}, \
    ERROR_HANDLING_CAPTURE \
    trace{trace}
