#pragma once

#include "cfgTools/boolSwitch.h"
#include "cfgTools/cfgSimpleString.h"
#include "history/historyObject.h"
#include "kits/moduleKit.h"
#include "profilerShell/profiler/profilerImpl.h"
#include "profilerShell/profilerReport/profilerReport.h"
#include "profilerShell/profilerTarget.h"
#include "timer/timer.h"

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

    using InitKit = KitCombine<ProfilerImpl::AllocKit, MsgLogKit>;
    using DeinitKit = KitCombine<ErrorLogKit, MsgLogKit>;
    using ProcessKit = KitCombine<ErrorLogKit, MsgLogExKit, MsgLogsKit, TimerKit>;
    using ReportKit = ProcessKit;

public:

    void serialize(const CfgSerializeKit& kit, bool hotkeys);

    stdbool init(stdPars(InitKit));
    void deinit();

public:

    inline void setDeviceControl(const ProfilerDeviceKit* deviceControl)
        {return profilerImpl.setDeviceControl(deviceControl);}

public:

    bool profilingActive() const {return profilerActive;}

    stdbool process(ProfilerTarget& target, float32 processingThroughput, stdPars(ProcessKit));


    stdbool makeReport(float32 processingThroughput, stdPars(ReportKit));

private:

    stdbool makeHtmlReport(float32 processingThroughput, stdPars(ReportKit));

private:

    void enterExternalScope();
    void leaveExternalScope();

private:

    BoolSwitch profilerActive{false};
    bool profilerActiveSteady = false;
    StandardSignal profilerResetSignal;

    ////

    StandardSignal htmlReportSignal;

    static CharArray htmlOutputDirName() {return STR("Output Directory");}
    SimpleStringVar htmlOutputDir{STR("")};

    profilerReport::HtmlReport htmlReport;

    ////

    static const Space profilerCapacity = 65536;
    ProfilerImpl profilerImpl;

    ////

    ProfilerScope externalScope;
    bool externalScopeIsEntered = false;

    ////

    uint32 cycleCount = 0;

    ////

    BoolSwitch displayFrameTime{false};
    static const Space frameTimeHistCapacity = 128;
    HistoryObjectStatic<float32, frameTimeHistCapacity> frameTimeHist;

};
