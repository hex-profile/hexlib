#include "configFile.h"

#include "cfg/cfgInterface.h"
#include "cfgVars/cfgOperations/cfgOperations.h"
#include "errorLog/errorLog.h"
#include "interfaces/fileTools.h"
#include "processTools/runAndWaitProcess.h"
#include "storage/classThunks.h"
#include "storage/constructDestruct.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"

namespace cfgVarsImpl {

//================================================================
//
// cfgvarsSynced
//
// Determines if any of cfgvars was changed
//
//================================================================

bool cfgvarsSynced(CfgSerialization& serialization)
{
    bool allSynced = true;

    auto visitVar = cfgVisitVar | [&] (auto& var)
    {
        check_flag(var.synced(), allSynced);
    };

    serialization({visitVar, CfgVisitSignalNull{}, CfgScopeVisitorNull{}});

    return allSynced;
}

//================================================================
//
// cfgvarsResetValue
//
// Sets all cfgvars to default values.
//
//================================================================

void cfgvarsResetValue(CfgSerialization& serialization)
{
    auto visitVar = cfgVisitVar | [&] (auto& var)
    {
        var.resetValue();
    };

    serialization({visitVar, CfgVisitSignalNull{}, CfgScopeVisitorNull{}});
}

//================================================================
//
// ConfigFileImpl
//
//================================================================

class ConfigFileImpl : public ConfigFile
{

public:

    virtual stdbool loadFile(const SimpleString& cfgFilename, stdPars(Kit));
    virtual void unloadFile();

    virtual stdbool loadVars(CfgSerialization& serialization, bool forceUpdate, stdPars(Kit));
    virtual stdbool saveVars(CfgSerialization& serialization, bool forceUpdate, bool& updateHappened, stdPars(Kit));

    virtual stdbool updateFile(bool forceUpdate, stdPars(Kit));
    virtual stdbool editFile(const SimpleString& configEditor, stdPars(Kit));

    virtual stdbool saveToString(StringReceiver& receiver, stdPars(Kit));
    virtual stdbool loadFromString(const CharArray& str, stdPars(Kit));

private:

    UniqueInstance<CfgOperations> operations;
    UniqueInstance<CfgTree> memory;

    bool memoryNotEqualToTheFile = false;

    SimpleString filename;
    bool updateFileEnabled = false;

};

//----------------------------------------------------------------

UniquePtr<ConfigFile> ConfigFile::create()
    {return makeUnique<ConfigFileImpl>();}

//================================================================
//
// GetSimpleString
//
//================================================================

class GetSimpleString : public fileTools::GetString
{

    SimpleString& str;

public:

    inline GetSimpleString(SimpleString& str)
        : str(str) {}

    bool operator()(const CharArray& value)
    {
        str = value;
        return def(str);
    }
};

//================================================================
//
// ConfigFileImpl::loadFile
//
//================================================================

stdbool ConfigFileImpl::loadFile(const SimpleString& cfgFilename, stdPars(Kit))
{
    memory->clearAll();

    filename.clear();
    updateFileEnabled = true;

    ////

    if (cfgFilename.size() == 0)
    {
        updateFileEnabled = false;
        returnFalse;
    }

    ////

    auto getFilename = fileTools::GetString::O | [&] (auto& str)
    {
        filename = str;
        return def(filename);
    };

    if_not
    (
        def(cfgFilename) &&
        fileTools::expandPath(cfgFilename.cstr(), getFilename) &&
        def(filename)
    )
    {
        printMsg(kit.msgLog, STR("Cannot get absolute path of config file '%0'"), cfgFilename.cstr(), msgWarn);
        updateFileEnabled = false;
        returnFalse;
    }

    ////

    if (fileTools::isFile(filename.cstr()))
    {
        if_not (errorBlock(operations->loadFromFile(*memory, filename.cstr(), false, stdPass)))
        {
            printMsg(kit.msgLog, STR("Config file %0 was not read successfully"), filename.cstr(), msgWarn);
            returnFalse;
        }
    }

    memoryNotEqualToTheFile = false;

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::unloadFile
//
//================================================================

void ConfigFileImpl::unloadFile()
{
    updateFileEnabled = false;
    filename.clear();
    memory->clearAll();
    memoryNotEqualToTheFile = false;
}

//================================================================
//
// ConfigFileImpl::saveVars
//
//================================================================

stdbool ConfigFileImpl::saveVars(CfgSerialization& serialization, bool forceUpdate, bool& updateHappened, stdPars(Kit))
{
    updateHappened = false;

    ////

    if (!forceUpdate && cfgvarsSynced(serialization))
        returnTrue;

    ////

    updateHappened = true;

    memoryNotEqualToTheFile = true;

    ////

    using namespace cfgOperations;

    SaveVarsOptions options{!forceUpdate, true};
    require(operations->saveVars(*memory, serialization, options, stdPass));

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::loadVars
//
//================================================================

stdbool ConfigFileImpl::loadVars(CfgSerialization& serialization, bool forceUpdate, stdPars(Kit))
{
    using namespace cfgOperations;

    LoadVarsOptions options{!forceUpdate, true};
    return operations->loadVars(*memory, serialization, options, stdPass);
}

//================================================================
//
// ConfigFileImpl::updateFile
//
//================================================================

stdbool ConfigFileImpl::updateFile(bool forceUpdate, stdPars(Kit))
{
    if_not (updateFileEnabled)
        returnFalse;

    if_not (forceUpdate || memoryNotEqualToTheFile)
        returnTrue;

    //
    // update file
    //

    if (errorBlock(operations->saveToFile(*memory, filename.cstr(), stdPass)))
    {
        memoryNotEqualToTheFile = false;
    }
    else
    {
        printMsg(kit.msgLog, STR("Cannot save config file %0, updating is stopped"), filename.cstr(), msgWarn);
        updateFileEnabled = false;
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// launchEditor
//
//================================================================

stdbool launchEditor(const SimpleString& configEditor, const SimpleString& filename, stdPars(MsgLogKit))
{
    SimpleString cmdLine; cmdLine << configEditor << CT(" \"") << filename << CT("\"");
    require(def(cmdLine));

    require(runAndWaitProcess(cmdLine.cstr(), stdPass));

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::editFile
//
//================================================================

stdbool ConfigFileImpl::editFile(const SimpleString& configEditor, stdPars(Kit))
{
    require(updateFileEnabled);
    REQUIRE(filename.size() != 0);

    ////

    require(updateFile(false, stdPass));

    ////

    if_not (errorBlock(launchEditor(configEditor, filename, stdPass)))
    {
        require(launchEditor(SimpleString{CT("notepad")}, filename, stdPass));
    }

    ////

    if_not (errorBlock(operations->loadFromFile(*memory, filename.cstr(), false, stdPass)))
    {
        printMsg(kit.msgLog, STR("Config file %0 was not read successfully"), filename.cstr(), msgWarn);
        returnFalse;
    }

    memoryNotEqualToTheFile = false;

    ////

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::saveToString
//
//================================================================

stdbool ConfigFileImpl::saveToString(StringReceiver& receiver, stdPars(Kit))
{
    require(operations->saveToString(*memory, receiver, stdPass));

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::loadFromString
//
//================================================================

stdbool ConfigFileImpl::loadFromString(const CharArray& str, stdPars(Kit))
{
    require(operations->loadFromString(*memory, str, stdPass));

    memoryNotEqualToTheFile = true;

    returnTrue;
}

//----------------------------------------------------------------

}
