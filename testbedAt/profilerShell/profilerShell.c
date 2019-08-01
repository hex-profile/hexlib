#include "profilerShell.h"

#include "profilerShell/profilerReport/profilerReport.h"
#include "profilerShell/profilerReport/quickReport.h"
#include "userOutput/msgLogKit.h"
#include "errorLog/errorLog.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "simpleString/simpleString.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// ProfilerShell::ProfilerShell
//
//================================================================

ProfilerShell::ProfilerShell()
{
    CharType* tempDir = getenv("TEMP");
  
    if (tempDir != 0)
        htmlOutputDir = SimpleString(tempDir) + SimpleString("\\profilerReport");
}

//================================================================
//
// ProfilerShell::serialize
//
//================================================================

void ProfilerShell::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Profiling");
        displayFrameTime.serialize(kit, STR("Display Frame Time"), STR("Ctrl+T"));
    
        profilerActiveSteady = profilerActive.serialize(kit, STR("Profiler Active"), STR("Alt+P"));

        profilerResetSignal.serialize(kit, STR("Profiler Reset"), STR("Ctrl+P"));

        if (profilerResetSignal) profilerActiveSteady = false;

        htmlReportSignal.serialize(kit, STR("Generate Report"), STR("P"));

        {
            CFG_NAMESPACE("HTML Report");
            htmlOutputDir.serialize(kit, htmlOutputDirName(), STR("Use double backslashes, for example C:\\\\Temp"));
            htmlReport.serialize(kit);
        }
    }
}

//================================================================
//
// ProfilerShell::init
//
//================================================================

stdbool ProfilerShell::init(stdPars(InitKit))
{
    deinit(); // deinit and reset to zero

    require(profilerImpl.realloc(profilerCapacity, stdPass));
    require(frameTimeHist.realloc(frameTimeHistCapacity, stdPass));

    returnTrue;
}

//================================================================
//
// ProfilerShell::deinit
//
//================================================================

void ProfilerShell::deinit()
{
    profilerImpl.dealloc();
 
    externalScopeIsEntered = false;
    cycleCount = 0;
}

//================================================================
//
// ProfilerShell::process
//
//================================================================

stdbool ProfilerShell::process(ProfilerTarget& target, float32 processingThroughput, stdPars(ProcessKit))
{
    //----------------------------------------------------------------
    //
    // Frame time report
    //
    //----------------------------------------------------------------

    breakBlock_
    {
        breakRequire(displayFrameTime);

        Space frameTimeCount = frameTimeHist.size();

        float32 totalTime = 0;
        Space totalCount = 0;

        for (Space k = 0; k < frameTimeCount; ++k)
        {
            float32* t = frameTimeHist[k];
            if (t == 0) break;

            totalTime += *t;
            totalCount += 1;

            if (totalTime >= 1.f) break;
        }

        bool ok = true;
        check_flag(totalCount >= 1, ok);
        check_flag(totalTime >= 1.f || totalCount >= 8, ok);

        float32 avgTime = totalTime / totalCount;

        printMsg(kit.localLog, STR("Frame time %0"), 
            !ok ? ParamMsg(paramMsg(STR("N/A"))) : 
            paramMsg(STR("%0 ms / %1 fps"), fltf(avgTime * 1e3, 2), fltf(1.f / avgTime, 1)));
    }

    //----------------------------------------------------------------
    //
    // Profiler switched?
    //
    //----------------------------------------------------------------

    bool profilerReset = !profilerActiveSteady && profilerActive;

    if (profilerReset)
    {
        profilerImpl.resetMemory();
        cycleCount = 0;
    }

    profilerActiveSteady = false;

    //----------------------------------------------------------------
    //
    // If profiling is not active, skip
    //
    //----------------------------------------------------------------

    if (profilerActive && profilerImpl.created())
    {
        if (profilerImpl.capacityExceeded())
            printMsgL(kit, STR("Profiling is active, PROFILER CAPACITY EXCEEDED!"), msgErr);
        else
            printMsg(kit.localLog, STR("Profiling is active"), profilerReset ? msgErr : msgWarn);
    }
    else
    {
        TimeMoment processBeg = kit.timer.moment();
        REMEMBER_CLEANUP(*frameTimeHist.add() = kit.timer.diff(processBeg, kit.timer.moment()));

        require(target.process(stdPassKit(ProfilerKit(nullptr))));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Leave external scope, if any
    //
    //----------------------------------------------------------------

    if (externalScopeIsEntered)
    {
        ProfilerThunk profilerThunk(profilerImpl);
        profilerThunk.leaveFunc(profilerThunk, externalScope);
    }

    externalScopeIsEntered = false;

    //----------------------------------------------------------------
    //
    // Monitor inner scope
    //
    //----------------------------------------------------------------

    bool processOk = false;

    {
        ProfilerThunk profilerThunk(profilerImpl);

        TimeMoment processBeg = kit.timer.moment();
        processOk = errorBlock(target.process(stdPassKit(ProfilerKit(&profilerThunk))));
        *frameTimeHist.add() = kit.timer.diff(processBeg, kit.timer.moment());

        CHECK(profilerImpl.checkResetScope());
        ++cycleCount;
    }

    //----------------------------------------------------------------
    //
    // Quick report
    //
    //----------------------------------------------------------------

    {
        CHECK(profilerImpl.checkResetScope());

        TimeMoment reportBegin = kit.timer.moment();
    
        auto kitEx = kitReplace(kit, MsgLogKit(kit.localLog));

        errorBlock(profilerQuickReport::namedNodesReport(profilerImpl.getRootNode(), profilerImpl.divTicksPerSec(), cycleCount, processingThroughput, stdPassKit(kitEx)));

        float32 reportTime = kit.timer.diff(reportBegin, kit.timer.moment());
        // printMsg(kit.localLog, STR("Profiler Quick Report %0 ms"), fltf(reportTime * 1e3f, 2), msgWarn);
    }

    //----------------------------------------------------------------
    //
    // HTML report
    //
    //----------------------------------------------------------------

    auto makeHtmlReport = [&] () -> stdbool
    {
        if_not (htmlOutputDir().length() > 0)
        {
            printMsgL(kit, STR("<%0> is not set"), htmlOutputDirName(), msgErr);
            returnFalse;
        }

        ////

        using namespace profilerReport;

        require(CHECK(profilerImpl.checkResetScope()));

        TimeMoment reportBegin = kit.timer.moment();
    
        auto kitEx = kitReplace(kit, MsgLogKit(kit.localLog));

        require(htmlReport.makeReport(MakeReportParams(profilerImpl.getRootNode(), profilerImpl.divTicksPerSec(), 
            cycleCount, processingThroughput, htmlOutputDir().cstr()), stdPassKit(kitEx)));

        float32 reportTime = kit.timer.diff(reportBegin, kit.timer.moment());

        printMsg(kit.localLog, STR("Profiler report saved to %0 (%1 ms)"), htmlOutputDir().cstr(), fltf(reportTime * 1e3f, 2), msgWarn);

        returnTrue;
    };

    ////

    if (htmlReportSignal != 0)
        errorBlock(makeHtmlReport());

    //----------------------------------------------------------------
    //
    // Enter external scope
    //
    //----------------------------------------------------------------

    if (monitorExternalScope)
    {
        ProfilerThunk profilerThunk(profilerImpl);
        externalScope.profiler = 0;
        profilerThunk.enterFunc(profilerThunk, externalScope, TRACE_AUTO_LOCATION_MSG("Profiler Outer Scope"));
        externalScopeIsEntered = true;
    }

    require(processOk);

    returnTrue;
}
