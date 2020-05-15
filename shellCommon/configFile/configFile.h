#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLogKit.h"
#include "cfg/cfgInterfaceFwd.h"
#include "timer/timer.h"
#include "simpleString/simpleString.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/fileTools.h"
#include "storage/dynamicClass.h"

namespace cfgVarsImpl {

//================================================================
//
// Utilities
//
//================================================================

bool cfgvarChanged(CfgSerialization& serialization);
void cfgvarClearChanged(CfgSerialization& serialization);
void cfgvarResetValue(CfgSerialization& serialization);

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
// CfgFileKit
//
//================================================================

KIT_COMBINE3(CfgFileKit, MsgLogKit, ErrorLogKit, FileToolsKit);

//================================================================
//
// ConfigFile
//
//----------------------------------------------------------------
//
// Config file full support
//
//================================================================

class ConfigFile
{

public:

    stdbool loadFile(const SimpleString& cfgFilename, stdPars(CfgFileKit));
    void unloadFile();

public:

    void loadVars(CfgSerialization& serialization);
    void saveVars(CfgSerialization& serialization, bool forceUpdate, bool* updateHappened = nullptr);

    stdbool updateFile(bool forceUpdate, stdPars(CfgFileKit));

public:

    stdbool editFile(const SimpleString& configEditor, stdPars(CfgFileKit));

public:

    ConfigFile();
    ~ConfigFile();

private:

    DynamicClass<class ConfigFileImpl> instance;

};

//----------------------------------------------------------------

}

using cfgVarsImpl::ConfigFile;
using cfgVarsImpl::CfgFileKit;
using cfgVarsImpl::ConfigUpdateDecimator;
using cfgVarsImpl::cfgvarChanged;
using cfgVarsImpl::cfgvarClearChanged;
using cfgVarsImpl::cfgvarResetValue;
