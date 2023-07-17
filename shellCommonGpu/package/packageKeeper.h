#pragma once

#include "storage/smartPtr.h"
#include "package/starterKit.h"
#include "cfg/cfgSerialization.h"
#include "minimalShell/minimalShellTypes.h"
#include "storage/adapters/lambdaThunk.h"

namespace packageImpl {
namespace packageKeeper {

using namespace minimalShell;

//================================================================
//
// PackageKeeper description.
//
//================================================================

/*
Holds internal state:
    - Minimal shell instance.
    - Engine memory state.
    - Formatter buffer
    - Signals histogram.
    - Config file.

Init helper:
    * Loads config.
    * Registers signals.
    * Initializes minimal shell.

Finalize helper:
    * Saves config.
    * Makes profiler report.

Process helper:
    * Gets a function params and StarterDebugKit.

    * Also gets a pointer to interface providing
        - Target process() function. The target function will get its parameters and EngineProcessKit.
        - Functions reallocValid() è realloc(), because the minimal shell will use them.
*/

//================================================================
//
// SerializeTarget
//
//================================================================

struct SerializeTarget
{
    virtual void serialize(const ModuleSerializeKit& kit) =0;
};

////

LAMBDA_THUNK
(
    serializeTargetThunk,
    SerializeTarget,
    void serialize(const ModuleSerializeKit& kit),
    lambda(kit)
)

//================================================================
//
// ProcessTarget
//
//================================================================

struct ProcessTarget : public SerializeTarget, public EngineModule
{
};

////

LAMBDA_THUNK4
(
    processTargetThunk,
    ProcessTarget,
    void serialize(const ModuleSerializeKit& kit),
    lambda0(kit),
    bool reallocValid() const,
    lambda1(),
    stdbool realloc(stdPars(EngineReallocKit)),
    lambda2(stdPassThru),
    stdbool process(stdPars(EngineProcessKit)),
    lambda3(stdPassThru)
)

//================================================================
//
// PackageKeeper
//
//================================================================

struct PackageKeeper
{
    static UniquePtr<PackageKeeper> create();
    virtual ~PackageKeeper() {}

    virtual void serialize(const CfgSerializeKit& kit) =0;
    virtual Settings& settings() =0;

    virtual stdbool init(const CharType* const configName, SerializeTarget& target, stdPars(StarterDebugKit)) =0;
    virtual stdbool finalize(SerializeTarget& target, stdPars(StarterDebugKit)) =0;
    virtual stdbool process(ProcessTarget& target, bool warmup, stdPars(StarterDebugKit)) =0;
};

//----------------------------------------------------------------

}

using packageKeeper::PackageKeeper;
using packageKeeper::processTargetThunk;
using packageKeeper::serializeTargetThunk;

}
