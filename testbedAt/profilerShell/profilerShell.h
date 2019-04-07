#pragma once

#include "profilerShell/profiler/profilerImpl.h"
#include "profilerShell/profilerTarget.h"
#include "cfgTools/boolSwitch.h"
#include "timer/timer.h"
#include "history/historyObj.h"
#include "kits/moduleKit.h"
#include "configFile/cfgSimpleString.h"
#include "interfaces/fileTools.h"
#include "profilerShell/profilerReport/profilerReport.h"

//================================================================
//
// ProfilerShell
//
// Startup-specific profiler support.
//
// Works via abstract 'profiler target' interface.
// The user should make an adapter of target object to this interface.
//
//================================================================

class ProfilerShell
{

public:

    KIT_COMBINE2(InitKit, ProfilerImpl::AllocKit, MsgLogKit);
    KIT_COMBINE2(DeinitKit, ErrorLogKit, MsgLogKit);
    KIT_COMBINE4(ProcessKit, ErrorLogKit, MsgLogsKit, TimerKit, FileToolsKit);

public:

    ProfilerShell();
    void serialize(const ModuleSerializeKit& kit);

    bool init(stdPars(InitKit));
    void deinit();

public:

    inline void setDeviceControl(const ProfilerDeviceKit* deviceControl)
        {return profilerImpl.setDeviceControl(deviceControl);}

public:

    bool process(ProfilerTarget& target, float32 processingThroughput, stdPars(ProcessKit));

private:

    BoolSwitch<false> profilerActive;
    bool profilerActiveSteady = false;
    StandardSignal profilerResetSignal;

    ////

    StandardSignal htmlReportSignal;
    static CharArray htmlOutputDirName() {return STR("Output Directory");}
    SimpleStringVar htmlOutputDir;
    profilerReport::HtmlReport htmlReport;

    ////

    static const Space profilerCapacity = 65536;
    ProfilerImpl profilerImpl;

    ////

    ProfilerScope externalScope;
    bool externalScopeIsEntered = false;
    bool monitorExternalScope = true;

    ////

    uint32 cycleCount = 0;

    ////

    BoolSwitch<true> displayFrameTime;
    static const Space frameTimeHistCapacity = 128;
    HistoryObjStatic<float32, frameTimeHistCapacity> frameTimeHist;

};
