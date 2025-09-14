#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "cfg/cfgSerialization.h"
#include "cfgVars/types/stringReceiver.h"
#include "errorLog/errorLogKit.h"
#include "simpleString/simpleString.h"
#include "stdFunc/stdFunc.h"
#include "storage/dynamicClass.h"
#include "storage/smartPtr.h"
#include "timer/timer.h"
#include "userOutput/msgLogKit.h"
#include "userOutput/printMsgTrace.h"
#include "userOutput/diagnosticKit.h"

namespace cfgVarsImpl {

//================================================================
//
// Utilities
//
//================================================================

bool cfgvarsSynced(CfgSerialization& serialization);
void cfgvarsResetValue(CfgSerialization& serialization);

//================================================================
//
// ConfigUpdateDecimator
//
//================================================================

class ConfigUpdateDecimator
{

public:

    bool shouldUpdate(Timer& timer)
    {
        TimeMoment currentTime = timer.moment();

        if (updateFileTimeInitialized && timer.diff(updateFileTime, currentTime) < updatePeriod)
            return false; // time is not passed, do not update

        updateFileTimeInitialized = true;
        updateFileTime = currentTime;

        return true;
    }

    void setPeriod(float32 period)
    {
        updatePeriod = period;
    }

private:

    float32 updatePeriod = 2.f;

    bool updateFileTimeInitialized = false;
    TimeMoment updateFileTime;

};

//================================================================
//
// Kit
//
//================================================================

using Kit = DiagnosticKit;

//================================================================
//
// ConfigFile
//
//----------------------------------------------------------------
//
// Config file full support
//
//================================================================

struct ConfigFile
{
    static UniquePtr<ConfigFile> create();
    virtual ~ConfigFile() {}

    ////

    virtual void loadFile(const SimpleString& cfgFilename, stdPars(Kit)) =0;
    virtual void unloadFile() =0;

    ////

    virtual void loadVars(CfgSerialization& serialization, bool forceUpdate, stdPars(Kit)) =0;

    virtual void saveVars(CfgSerialization& serialization, bool forceUpdate, bool& updateHappened, stdPars(Kit)) =0;

    sysinline void saveVars(CfgSerialization& serialization, bool forceUpdate, stdPars(Kit))
    {
        bool updateHappened{};
        saveVars(serialization, forceUpdate, updateHappened, stdPassThru);
    }

    ////

    virtual void updateFile(bool forceUpdate, stdPars(Kit)) =0;
    virtual void editFile(const SimpleString& configEditor, stdPars(Kit)) =0;

    ////

    virtual void saveToString(StringReceiver& receiver, stdPars(Kit)) =0;
    virtual void loadFromString(const CharArray& str, stdPars(Kit)) =0;
};

//----------------------------------------------------------------

}

using cfgVarsImpl::ConfigFile;
using cfgVarsImpl::ConfigUpdateDecimator;
using cfgVarsImpl::cfgvarsSynced;
using cfgVarsImpl::cfgvarsResetValue;
