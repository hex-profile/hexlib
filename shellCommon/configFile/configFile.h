#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "storage/staticClass.h"
#include "userOutput/msgLogKit.h"
#include "cfg/cfgInterfaceFwd.h"
#include "timer/timer.h"
#include "simpleString/simpleString.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/fileTools.h"

namespace cfgVarsImpl {

//================================================================
//
// Utilities
//
//================================================================

bool cfgvarChanged(CfgSerialization& serialization);
void cfgvarClearChanged(CfgSerialization& serialization);

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

        if
        (
            updateFileTimeInitialized
            && timer.diff(updateFileTime, currentTime) < 2.0f
        )
            return false; // time is not passed, do not update

        updateFileTimeInitialized = true;
        updateFileTime = currentTime;

        return true;
    }

private:

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

    bool loadFile(const SimpleString& cfgFilename, stdPars(CfgFileKit));
    void unloadFile();

public:

    void loadVars(CfgSerialization& serialization);
    void saveVars(CfgSerialization& serialization, bool forceUpdate);

    void updateFile(bool forceUpdate, stdPars(CfgFileKit));

public:

    void editFile(const SimpleString& configEditor, stdPars(CfgFileKit));

public:

    ConfigFile();
    ~ConfigFile();

private:

    StaticClass<class ConfigFileImpl, 256> instance;

};

//----------------------------------------------------------------

}

using cfgVarsImpl::ConfigFile;
using cfgVarsImpl::CfgFileKit;
using cfgVarsImpl::ConfigUpdateDecimator;
using cfgVarsImpl::cfgvarChanged;
using cfgVarsImpl::cfgvarClearChanged;
