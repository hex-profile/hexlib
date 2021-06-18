#pragma once

#include "storage/smartPtr.h"
#include "package/starterKit.h"
#include "configFile/cfgSerialization.h"
#include "minimalShell/minimalShellTypes.h"

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

//================================================================
//
// ProcessTarget
//
//================================================================

struct ProcessTarget : public SerializeTarget, public EngineModule
{
};

//================================================================
//
// SerializeTargetMaker
//
//================================================================

template <typename SerializeFunc>
class SerializeTargetMaker : public SerializeTarget
{

public:

    SerializeTargetMaker(const SerializeFunc& serializeFunc)
        : serializeFunc{serializeFunc} {}

    virtual void serialize(const ModuleSerializeKit& kit)
        {return serializeFunc(kit);}

private:
    
    SerializeFunc const serializeFunc;

};

//----------------------------------------------------------------

template <typename SerializeFunc>
inline auto serializeTargetMaker(const SerializeFunc& serializeFunc)
{
    return SerializeTargetMaker<SerializeFunc>{serializeFunc};
}

//================================================================
//
// ProcessTargetMaker
//
//================================================================

template <typename SerializeFunc, typename ReallocValidFunc, typename ReallocFunc, typename ProcessFunc>
class ProcessTargetMaker : public ProcessTarget
{

public:

    ProcessTargetMaker
    (
        const SerializeFunc& serializeFunc,
        const ReallocValidFunc& reallocValidFunc,
        const ReallocFunc& reallocFunc,
        const ProcessFunc& processFunc
    )
        : 
        serializeFunc{serializeFunc},
        reallocValidFunc{reallocValidFunc},
        reallocFunc{reallocFunc},
        processFunc{processFunc} 
    {
    }

    virtual void serialize(const ModuleSerializeKit& kit)
        {return serializeFunc(kit);}

    virtual bool reallocValid() const
        {return reallocValidFunc();}

    virtual stdbool realloc(stdPars(EngineReallocKit))
        {return reallocFunc(stdPassThru);}

    virtual stdbool process(stdPars(EngineProcessKit))
        {return processFunc(stdPassThru);}

private:
    
    SerializeFunc const serializeFunc;
    ReallocValidFunc const reallocValidFunc;
    ReallocFunc const reallocFunc;
    ProcessFunc const processFunc;

};

//----------------------------------------------------------------

template <typename SerializeFunc, typename ReallocValidFunc, typename ReallocFunc, typename ProcessFunc>
inline ProcessTargetMaker<SerializeFunc, ReallocValidFunc, ReallocFunc, ProcessFunc> processTargetMaker
(
    const SerializeFunc& serializeFunc,
    const ReallocValidFunc& reallocValidFunc,
    const ReallocFunc& reallocFunc,
    const ProcessFunc& processFunc
)
{
    return {serializeFunc, reallocValidFunc, reallocFunc, processFunc};
}

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
using packageKeeper::processTargetMaker;
using packageKeeper::serializeTargetMaker;

}
