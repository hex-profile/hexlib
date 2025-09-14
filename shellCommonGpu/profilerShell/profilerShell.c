#include "profilerShell.h"

#include "dataAlloc/arrayObjectMemory.inl"
#include "errorLog/errorLog.h"
#include "profilerShell/profilerReport/profilerReport.h"
#include "profilerShell/profilerReport/quickReport.h"
#include "simpleString/simpleString.h"
#include "storage/rememberCleanup.h"
#include "userOutput/msgLogKit.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// ProfilerShell::serialize
//
//================================================================

void ProfilerShell::serialize(const CfgSerializeKit& kit, bool hotkeys)
{
    #define HOTSTR(s) (!hotkeys ? STR("") : STR(s))

    {
        CFG_NAMESPACE("Profiling");

        ftmDisplayed.serialize(kit, STR("Frame Time Info"), HOTSTR("Ctrl+T"));
        ftmHalfLife.serialize(kit, STR("Frame Time Smoothing Period"), STR("In seconds"));

        profilerActiveSteady = profilerActive.serialize(kit, STR("Profiler Active"), HOTSTR("Alt+P"));

        profilerResetSignal.serialize(kit, STR("Profiler Reset"), HOTSTR("Ctrl+P"));

        if (profilerResetSignal) profilerActiveSteady = false;

        htmlReportSignal.serialize(kit, STR("Generate Report"), HOTSTR("P"));

        {
            CFG_NAMESPACE("HTML Report");
            htmlOutputDir.serialize(kit, htmlOutputDirName(), HOTSTR("For example, C:/Temp"));
            htmlReport.serialize(kit);
        }
    }

    #undef HOTSTR
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
// ProfilerShell::makeHtmlReport
//
//================================================================

stdbool ProfilerShell::makeHtmlReport(float32 processingThroughput, stdPars(ReportKit))
{
    SimpleString outputDir{htmlOutputDir()};
    REQUIRE(def(outputDir));

    ////

    if (outputDir.size() == 0)
    {
        auto tempDir = getenv("HEXLIB_OUTPUT");

        if_not (tempDir)
            tempDir = getenv("TEMP");

        if (tempDir != 0)
            outputDir.clear() << tempDir << "/profilerReport";

        REQUIRE(def(outputDir));
    }

    ////

    if (outputDir.size() == 0)
    {
        printMsgL(kit, STR("<%0> is not set"), htmlOutputDirName(), msgErr);
        returnFalse;
    }

    ////

    using namespace profilerReport;

    require(CHECK(profilerImpl.checkResetScope()));

    TimeMoment reportBegin = kit.timer.moment();

    auto kitEx = kit; // kitReplace(kit, MsgLogKit(kit.localLog));

    require(htmlReport.makeReport(MakeReportParams{profilerImpl.getRootNode(), profilerImpl.divTicksPerSec(),
        cycleCount, processingThroughput, outputDir.cstr()}, stdPassKit(kitEx)));

    float32 reportTime = kit.timer.diff(reportBegin, kit.timer.moment());

    printMsg(kit.localLog, STR("Profiler report saved to %0 (%1 ms)"), outputDir, fltf(reportTime * 1e3f, 2), msgWarn);

    returnTrue;
}

//================================================================
//
// ProfilerShell::enterExternalScope
// ProfilerShell::leaveExternalScope
//
//================================================================

void ProfilerShell::enterExternalScope()
{
    if_not (externalScopeIsEntered)
    {
        ProfilerThunk profilerThunk(profilerImpl);
        externalScope.profiler = 0;
        profilerThunk.enterFunc(profilerThunk, externalScope, TRACE_AUTO_LOCATION_MSG("Profiler External Scope"));
        externalScopeIsEntered = true;
    }
}

//----------------------------------------------------------------

void ProfilerShell::leaveExternalScope()
{
    if (externalScopeIsEntered)
    {
        ProfilerThunk profilerThunk(profilerImpl);
        profilerThunk.leaveFunc(profilerThunk, externalScope);
        externalScopeIsEntered = false;
    }
}

//================================================================
//
// ProfilerShell::ftmUpdate
//
//================================================================

stdbool ProfilerShell::ftmUpdate(float32 frameTime, stdPars(ErrorLogKit))
{
    REQUIRE(frameTime >= 0);

    ////

    ftmLastTime = frameTime;

    ////

    auto periods = clampMin(convertUp<Space>(frameTime * ftmDivResolutionPeriod), 1);
    auto actualPeriod = frameTime * fastRecip(float32(periods));
    auto weight = clampMax(actualPeriod * ftmDivResolutionPeriod, 1.f);

    ////

    auto temporalFactor = tpf::TemporalFactor(ftmHalfLife * ftmDivResolutionPeriod, ftmStages);

    for_count (i, periods)
        ftmFilter.add(weight, frameTime, temporalFactor);

    returnTrue;
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

    auto frameTimeReport = [&] ()
    {
        if_not (ftmDisplayed)
            returnTrue;

        float32 avgTime = ftmFilter();

        printMsg(kit.localLog, STR("% fps, cycle % ms, last % ms"),
            fltf(1.f / avgTime, 1), fltf(avgTime * 1e3, 1), fltf(ftmLastTime * 1e3, 1));

        returnTrue;
    };

    ////

    errorBlock(frameTimeReport());

    //----------------------------------------------------------------
    //
    // Profiling toggled?
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
        TimeMoment processStart = kit.timer.moment();
        REMEMBER_CLEANUP(errorBlock(ftmUpdate(kit.timer.diff(processStart, kit.timer.moment()), stdPass)));

        require(target(stdPassKit(ProfilerKit(nullptr))));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Leave external scope, if any
    //
    //----------------------------------------------------------------

    leaveExternalScope();

    //----------------------------------------------------------------
    //
    // Monitor inner scope
    //
    //----------------------------------------------------------------

    bool processOk = false;

    {
        ProfilerThunk profilerThunk(profilerImpl);

        TimeMoment processStart = kit.timer.moment();
        REMEMBER_CLEANUP(errorBlock(ftmUpdate(kit.timer.diff(processStart, kit.timer.moment()), stdPass)));

        processOk = errorBlock(target(stdPassKit(ProfilerKit(&profilerThunk))));

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

        auto kitEx = kitReplace(kit, MsgLogKit(kit.localLog));

        errorBlock(profilerQuickReport::namedNodesReport(profilerImpl.getRootNode(), profilerImpl.divTicksPerSec(), cycleCount, processingThroughput, stdPassKit(kitEx)));
    }

    //----------------------------------------------------------------
    //
    // HTML report
    //
    //----------------------------------------------------------------

    if (htmlReportSignal != 0)
        errorBlock(makeHtmlReport(processingThroughput, stdPass));

    //----------------------------------------------------------------
    //
    // Enter external scope
    //
    //----------------------------------------------------------------

    enterExternalScope();

    ////

    require(processOk);

    returnTrue;
}

//================================================================
//
// ProfilerShell::makeReport
//
//================================================================

stdbool ProfilerShell::makeReport(float32 processingThroughput, stdPars(ReportKit))
{
    leaveExternalScope();
    REMEMBER_CLEANUP(enterExternalScope());
    return makeHtmlReport(processingThroughput, stdPass);
}
