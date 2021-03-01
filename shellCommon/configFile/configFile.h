#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "configFile/cfgSerialization.h"
#include "configFile/stringReceiver.h"
#include "errorLog/errorLogKit.h"
#include "interfaces/fileToolsKit.h"
#include "simpleString/simpleString.h"
#include "stdFunc/stdFunc.h"
#include "storage/dynamicClass.h"
#include "storage/smartPtr.h"
#include "timer/timer.h"
#include "userOutput/msgLogKit.h"
#include "userOutput/errorLogEx.h"

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
// Kit
//
//================================================================

using Kit = KitCombine<MsgLogKit, ErrorLogKit, ErrorLogExKit, FileToolsKit>;

//================================================================
//
// stringReceiverByLambda
//
//================================================================

template <typename Lambda>
class StringReceiverByLambda : public StringReceiver
{

public:

    StringReceiverByLambda(const Lambda& lambda)
        : lambda{lambda} {}

    virtual stdbool receive(const CharArray& str, stdNullPars)
        {return lambda(str, stdNullPass);}

private:

    Lambda lambda;

};

//----------------------------------------------------------------

template <typename Lambda>
inline auto stringReceiverByLambda(const Lambda& lambda)
    {return StringReceiverByLambda<Lambda>{lambda};}

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

    virtual stdbool loadFile(const SimpleString& cfgFilename, stdPars(Kit)) =0;
    virtual void unloadFile() =0;

    virtual void loadVars(CfgSerialization& serialization) =0;
    virtual void saveVars(CfgSerialization& serialization, bool forceUpdate, bool* updateHappened = nullptr) =0;

    virtual stdbool updateFile(bool forceUpdate, stdPars(Kit)) =0;
    virtual stdbool editFile(const SimpleString& configEditor, stdPars(Kit)) =0;

    virtual stdbool saveToString(StringReceiver& receiver, stdPars(Kit)) =0;
    virtual stdbool loadFromString(const CharArray& str, stdPars(Kit)) =0;
};

//----------------------------------------------------------------

}

using cfgVarsImpl::ConfigFile;
using cfgVarsImpl::ConfigUpdateDecimator;
using cfgVarsImpl::cfgvarChanged;
using cfgVarsImpl::cfgvarClearChanged;
using cfgVarsImpl::cfgvarResetValue;
