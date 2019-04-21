#include "configFile.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include "storage/rememberCleanup.h"
#include "configFile/cfgSerializeImpl.h"
#include "configFile/cfgFileEnv.h"
#include "storage/constructDestruct.h"
#include "storage/classThunks.h"
#include "cfg/cfgInterface.h"
#include "userOutput/printMsg.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgEx.h"

namespace cfgVarsImpl {

//================================================================
//
// ProcessTools
//
//================================================================

struct ProcessTools
{
    virtual bool runAndWaitProcess(const CharType* cmdLine) =0;
};

//================================================================
//
// ProcessToolImpl (Win32)
//
//================================================================

#if defined(_WIN32)

class ProcessToolImplThunk : public ProcessTools
{

public:

    bool runAndWaitProcess(const CharType* cmdLine)
    {
        STARTUPINFO si;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);

        PROCESS_INFORMATION pi;
        ZeroMemory(&pi, sizeof(pi));

        if_not (CreateProcess(NULL, (CharType*) cmdLine, NULL, NULL, false, 0, NULL, NULL, &si, &pi) != 0)
        {
            printMsg(msgLog, STR("Cannot launch: <%0>, error code %1"), cmdLine, hex(uint32(GetLastError())), msgErr);
            msgLog.update();
            return false;
        }

        REMEMBER_CLEANUP1(CloseHandle(pi.hProcess), PROCESS_INFORMATION&, pi);
        REMEMBER_CLEANUP1(CloseHandle(pi.hThread), PROCESS_INFORMATION&, pi);

        // Wait process
        WaitForSingleObject(pi.hProcess, INFINITE);

        return true;
    }

    inline ProcessToolImplThunk(MsgLog& msgLog)
        : msgLog(msgLog) {}

private:

    MsgLog& msgLog;

};

//================================================================
//
// ProcessToolImpl (not implemented)
//
//================================================================

#else

class ProcessToolImplThunk : public ProcessTools
{

public:

    bool runAndWaitProcess(const CharType* cmdLine)
    {
        printMsg(msgLog, STR("Not implemented"), msgErr);
        return false;
    }

    inline ProcessToolImplThunk(MsgLog& msgLog)
        : msgLog(msgLog) {}

private:

    MsgLog& msgLog;

};

#endif

//================================================================
//
// cfgvarChanged
//
// Determines if any of cfgvars was changed
//
//================================================================

struct CheckCfgvarsChanged : public CfgVisitor
{
    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
    {
        if (var.changed())
            anyChange = true;
    }

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
        {}

    bool anyChange = false;
};

//----------------------------------------------------------------

bool cfgvarChanged(CfgSerialization& serialization)
{
    CheckCfgvarsChanged checkChanged;
    serialization.serialize(CfgSerializeKit(checkChanged, 0));
    return checkChanged.anyChange;
}

//================================================================
//
// cfgvarsClearChanged
//
// Clears dirty flag for all cfgvars
//
//================================================================

struct CfgvarClearChanged : public CfgVisitor
{
    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
        {var.clearChanged();}

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
        {}
};

//----------------------------------------------------------------

void cfgvarClearChanged(CfgSerialization& serialization)
{
    CfgvarClearChanged clearChanged;
    CfgSerializeKit kit(clearChanged, 0);
    serialization.serialize(kit);
}

//================================================================
//
// ConfigFileImpl
//
//================================================================

class ConfigFileImpl
{

public:

    stdbool loadFile(const SimpleString& cfgFilename, stdPars(CfgFileKit));
    void unloadFile();

    void loadVars(CfgSerialization& serialization);
    void saveVars(CfgSerialization& serialization, bool forceUpdate);

    stdvoid updateFile(bool forceUpdate, stdPars(CfgFileKit));

    stdvoid editFile(const SimpleString& configEditor, stdPars(CfgFileKit));

private:

    FileEnvSTL memory;
    bool memoryChanged = false;

    SimpleString filename;
    bool updateFileEnabled = false;

};

//================================================================
//
// GetSimpleString
//
//================================================================

class GetSimpleString : public GetString
{

    SimpleString& str;

public:

    inline GetSimpleString(SimpleString& str)
        : str(str) {}

    bool setBuffer(const CharType* bufArray, size_t bufSize)
    {
        str.assign(bufArray, bufSize);
        return str.ok();
    }
};

//================================================================
//
// ConfigFileImpl::loadFile
//
//================================================================

stdbool ConfigFileImpl::loadFile(const SimpleString& cfgFilename, stdPars(CfgFileKit))
{
    stdBegin;

    ////

    memory.eraseAll();
    filename.clear();
    updateFileEnabled = true;

    ////

    if (cfgFilename.length() == 0)
    {
        updateFileEnabled = false;
        return false;
    }

    ////

    GetSimpleString getFilename(filename);

    if_not
    (
        cfgFilename.ok() &&
        kit.fileTools.expandPath(cfgFilename.cstr(), getFilename) &&
        filename.ok()
    )
    {
        printMsg(kit.msgLog, STR("Cannot get absolute path of config file '%0'"), cfgFilename.cstr(), msgWarn);
        updateFileEnabled = false;
        return false;
    }

    ////

    if_not (memory.loadFromFile(filename.cstr(), kit.fileTools, stdPass))
    {
        printMsg(kit.msgLog, STR("Config file %0 was not read successfully"), filename.cstr(), msgWarn);
        return false;
    }

    memoryChanged = false;

    stdEnd;
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
    memory.eraseAll();
    memoryChanged = false;
}

//================================================================
//
// ConfigFileImpl::loadVars
//
//================================================================

void ConfigFileImpl::loadVars(CfgSerialization& serialization)
{
    loadVarsFromStringEnv(serialization, 0, memory);
}

//================================================================
//
// ConfigFileImpl::saveVars
//
//================================================================

void ConfigFileImpl::saveVars(CfgSerialization& serialization, bool forceUpdate)
{
    if (forceUpdate || cfgvarChanged(serialization))
    {
        saveVarsToStringEnv(serialization, 0, memory);
        cfgvarClearChanged(serialization);
        memoryChanged = true;
    }
}

//================================================================
//
// ConfigFileImpl::updateFile
//
//================================================================

stdvoid ConfigFileImpl::updateFile(bool forceUpdate, stdPars(CfgFileKit))
{
    stdBegin;

    if_not (updateFileEnabled)
        return;

    if_not (forceUpdate || memoryChanged)
        return;

    //
    // update file
    //

    if (memory.saveToFile(filename.cstr(), kit.fileTools, stdPass))
    {
        memoryChanged = false;
    }
    else
    {
        printMsg(kit.msgLog, STR("Cannot save config file %0, updating is stopped"), filename.cstr(), msgWarn);
        updateFileEnabled = false;
        return;
    }

    stdEndv;
}

//================================================================
//
// launchEditor
//
//================================================================

stdbool launchEditor(const SimpleString& configEditor, const SimpleString& filename, stdPars(MsgLogKit))
{
    SimpleString cmdLine = configEditor + CT(" \"") + filename + CT("\"");
    if_not (cmdLine.ok()) return false;

    ProcessToolImplThunk processTool(kit.msgLog);
    require(processTool.runAndWaitProcess(cmdLine.cstr()));

    return true;
}

//================================================================
//
// ConfigFileImpl::editFile
//
//================================================================

stdvoid ConfigFileImpl::editFile(const SimpleString& configEditor, stdPars(CfgFileKit))
{
    stdBegin;

    requirev(updateFileEnabled);
    REQUIREV(filename.length() != 0);

    ////

    updateFile(false, stdPass);

    ////

    if_not (launchEditor(configEditor, filename, stdPass))
    {
        if_not (launchEditor(CT("notepad"), filename, stdPass))
            return;
    }

    if_not (memory.loadFromFile(filename.cstr(), kit.fileTools, stdPass))
        printMsg(kit.msgLog, STR("Config file %0 was not read successfully"), filename.cstr(), msgWarn);

    memoryChanged = false;

    ////

    stdEndv;
}

//================================================================
//
// ConfigFile::*
//
//================================================================

ConfigFile::ConfigFile()
    {}

ConfigFile::~ConfigFile()
    {}

stdbool ConfigFile::loadFile(const SimpleString& cfgFilename, stdPars(CfgFileKit))
    {return instance->loadFile(cfgFilename, stdPassThru);}

void ConfigFile::unloadFile()
    {instance->unloadFile();}

void ConfigFile::loadVars(CfgSerialization& serialization)
    {instance->loadVars(serialization);}

void ConfigFile::saveVars(CfgSerialization& serialization, bool forceUpdate)
    {instance->saveVars(serialization, forceUpdate);}

stdvoid ConfigFile::updateFile(bool forceUpdate, stdPars(CfgFileKit))
    {instance->updateFile(forceUpdate, stdPassThru);}

stdvoid ConfigFile::editFile(const SimpleString& configEditor, stdPars(CfgFileKit))
    {instance->editFile(configEditor, stdPassThru);}

//----------------------------------------------------------------

}
