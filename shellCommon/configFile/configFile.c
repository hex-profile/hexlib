#include "configFile.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
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
#include "interfaces/fileTools.h"

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
    CheckCfgvarsChanged visitor;
    serialization.serialize(CfgSerializeKit(visitor, nullptr));
    return visitor.anyChange;
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
    CfgvarClearChanged visitor;
    CfgSerializeKit kit(visitor, 0);
    serialization.serialize(kit);
}

//================================================================
//
// cfgvarsClearChanged
//
// Sets all cfgvars to default values.
//
//================================================================

struct CfgvarSetDefaultValue : public CfgVisitor
{
    void operator()(const CfgNamespace* scope, const CfgSerializeVariable& var)
        {var.resetValue();}

    void operator()(const CfgNamespace* scope, const CfgSerializeSignal& signal)
        {}
};

//----------------------------------------------------------------

void cfgvarResetValue(CfgSerialization& serialization)
{
    CfgvarSetDefaultValue visitor;
    CfgSerializeKit kit(visitor, 0);
    serialization.serialize(kit);
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

    virtual void loadVars(CfgSerialization& serialization);
    virtual void saveVars(CfgSerialization& serialization, bool forceUpdate, bool* updateHappened);

    virtual stdbool updateFile(bool forceUpdate, stdPars(Kit));
    virtual stdbool editFile(const SimpleString& configEditor, stdPars(Kit));

    virtual stdbool saveToString(StringReceiver& receiver, stdPars(Kit));
    virtual stdbool loadFromString(const CharArray& str, stdPars(Kit));

private:

    UniquePtr<FileEnvSTL> memoryPtr = FileEnvSTL::create();
    FileEnvSTL& memory = *memoryPtr;

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

class GetSimpleString : public GetString
{

    SimpleString& str;

public:

    inline GetSimpleString(SimpleString& str)
        : str(str) {}

    bool setBuffer(const CharType* bufArray, size_t bufSize)
    {
        str.assign(bufArray, bufSize);
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
    memory.eraseAll();
    filename.clear();
    updateFileEnabled = true;

    ////

    if (cfgFilename.size() == 0)
    {
        updateFileEnabled = false;
        returnFalse;
    }

    ////

    GetSimpleString getFilename(filename);

    if_not
    (
        def(cfgFilename) &&
        kit.fileTools.expandPath(cfgFilename.cstr(), getFilename) &&
        def(filename)
    )
    {
        printMsg(kit.msgLog, STR("Cannot get absolute path of config file '%0'"), cfgFilename.cstr(), msgWarn);
        updateFileEnabled = false;
        returnFalse;
    }

    ////

    if_not (errorBlock(memory.loadFromFile(filename.cstr(), stdPass)))
    {
        printMsg(kit.msgLog, STR("Config file %0 was not read successfully"), filename.cstr(), msgWarn);
        returnFalse;
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
    memory.eraseAll();
    memoryNotEqualToTheFile = false;
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

void ConfigFileImpl::saveVars(CfgSerialization& serialization, bool forceUpdate, bool* updateHappened)
{
    if (updateHappened) 
        *updateHappened = false;

    if (forceUpdate || cfgvarChanged(serialization))
    {
        if (updateHappened) 
            *updateHappened = true;

        saveVarsToStringEnv(serialization, 0, memory);
        cfgvarClearChanged(serialization);

        memoryNotEqualToTheFile = true;
    }
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

    if (errorBlock(memory.saveToFile(filename.cstr(), stdPass)))
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

    ProcessToolImplThunk processTool(kit.msgLog);
    require(processTool.runAndWaitProcess(cmdLine.cstr()));

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

    if_not (errorBlock(memory.loadFromFile(filename.cstr(), stdPass)))
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
    require(memory.saveToString(receiver, stdPass));

    returnTrue;
}

//================================================================
//
// ConfigFileImpl::loadFromString
//
//================================================================

stdbool ConfigFileImpl::loadFromString(const CharArray& str, stdPars(Kit))
{
    require(memory.loadFromString(str, stdPass));

    memoryNotEqualToTheFile = true;

    returnTrue;
}

//----------------------------------------------------------------

}
