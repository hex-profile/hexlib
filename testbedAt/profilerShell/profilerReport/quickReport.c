#include "quickReport.h"

#include "numbers/mathIntrinsics.h"
#include "userOutput/printMsg.h"
#include "userOutput/paramMsg.h"
#include "formatting/prettyNumber.h"
#include "errorLog/errorLog.h"

namespace profilerQuickReport {

//================================================================
//
// NodeVisitor
//
//================================================================

struct NodeVisitor
{
    virtual void operator()
    (
        int32 treeLevel,
        const CharArray& userName,
        const TraceScope& scope,
        bool pureTime,
        float32 timeFraction,
        float32 avgPerCycleCount,
        float32 avgNormalTime,
        float32 avgDeviceTime,
        float32 avgElemCount,
        float32 avgElemNormalTime,
        float32 avgElemDeviceTime
    )
    =0;
};

//================================================================
//
// DisplayReportParams
//
//================================================================

KIT_CREATE4(
    DisplayReportParams,
    float32, tickFactor,
    uint32, cycleCount,
    float32, fullRunTime,
    NodeVisitor&, visitor
);

//================================================================
//
// displayProfilerTree
//
//================================================================

stdbool displayProfilerTree(ProfilerNode& node, const DisplayReportParams& o, Space treeLevel, const TraceScope* nodeScope, stdPars(ErrorLogKit))
{
    //----------------------------------------------------------------
    //
    // This node
    //
    //----------------------------------------------------------------

    TraceScope nodeStack(node.location, nodeScope);

    if (node.userName.size != 0)
    {
        float32 divCycleCount = 1.f / clampMin(o.cycleCount, 1U);
        float32 avgPerCycleCount = float32(node.counter) * divCycleCount;

        ////

        float32 nodeTotalTime = float32(node.totalTimeSum) * o.tickFactor;
        float32 nodeDeviceTime = float32(node.deviceTotalTime);

        ////

        float32 divNodeCounter = nativeRecipZero(float32(node.counter));

        float32 avgNormalTime = nodeTotalTime * divNodeCounter;
        float32 avgDeviceTime = nodeDeviceTime * divNodeCounter;

        ////

        float32 avgElemCount = float32(node.totalElemCount) * divNodeCounter;

        ////

        float32 divNodeTotalElemCount = nativeRecipZero(float32(node.totalElemCount));

        float32 avgElemNormalTime = nodeTotalTime * divNodeTotalElemCount;
        float32 avgElemDeviceTime = nodeDeviceTime * divNodeTotalElemCount;

        ////

        float32 nodeTimeFraction = nodeTotalTime / o.fullRunTime;

        o.visitor(treeLevel, node.userName, nodeStack, false, nodeTimeFraction, avgPerCycleCount, avgNormalTime, avgDeviceTime, avgElemCount, 
            avgElemNormalTime, avgElemDeviceTime);
    }
  
    //----------------------------------------------------------------
    //
    // Fast way
    //
    //----------------------------------------------------------------

    for (ProfilerNode* p = node.lastChild; p != 0; p = p->prevBrother)
        require(displayProfilerTree(*p, o, treeLevel + 1, &nodeStack, stdPass));

    ////

    returnTrue;
}

//================================================================
//
// PrintNode
//
//================================================================

class PrintNode : public NodeVisitor
{

public:

    void operator()
    (
        int32 treeLevel,
        const CharArray& userName,
        const TraceScope& scope,
        bool pureTime,
        float32 timeFraction,
        float32 avgPerCycleCount,
        float32 avgNormalTime,
        float32 avgDeviceTime,
        float32 avgElemCount,
        float32 avgElemNormalTime,
        float32 avgElemDeviceTime
    )
    {

        float32 timePercentage = timeFraction * 100;

        ////

        float32 deviceFraction = avgDeviceTime / avgNormalTime;
        if_not (def(deviceFraction)) deviceFraction = avgDeviceTime ? 1.f : 0.f;

        bool deviceMode = (deviceFraction != 0);

        float32 avgTimeMs = (deviceMode ? avgDeviceTime : avgNormalTime) * 1000;
        float32 avgElemTime = (deviceMode ? avgElemDeviceTime : avgElemNormalTime);

        ////

        bool perCycleCountImportant = convertNearest<Space>(avgPerCycleCount * 10.f) != 10;

        ////

        printMsg
        (
            kit.msgLog,
            STR("%0%1: %2 is %4%5, %3%% total"),
            STR(""), // indent
            userName.size ? userName : charArrayFromPtr(scope.location), // 1
            paramMsg(deviceMode ? STR("device %0%%") : (pureTime ? STR("body") : STR("time")), fltf(deviceFraction * 1e2f, 0)), // 2
            fltf(timePercentage, 1), // 3
            paramMsg(!perCycleCountImportant ? STR("%0 ms") : STR("%0 ms x %1 fraction"), fltf(avgTimeMs, 2), fltf(avgPerCycleCount, 1)), // 4
            paramMsg(avgElemTime == 0 ? STR("") : STR(" (%0 clocks/elem at %1 square)"), prettyNumber(fltg(avgElemTime * processingThroughput, 3)), fltf(sqrtf(avgElemCount), 0))
        );

        ////

        if (0)
        {
            if (scope.prev)
            {
                for (const TraceScope* p = scope.prev; p != 0; p = p->prev)
                    printMsg(kit.msgLog, STR("    %0: called from"), p->location);

                printMsg(kit.msgLog, STR(""));
            }
        }

    }

public:

    inline PrintNode(float32 processingThroughput, const ReportKit& kit)
        : processingThroughput(processingThroughput), kit(kit) {}

private:

    float32 processingThroughput;
    ReportKit kit;

};

//================================================================
//
// namedNodesReport
//
//================================================================

stdbool namedNodesReport
(
    ProfilerNode* rootNode, 
    float32 divTicksPerSec, 
    uint32 cycleCount, 
    float32 processingThroughput,
    stdPars(ReportKit)
)
{
    REQUIRE(cycleCount >= 0);
    require(cycleCount >= 1);

    //
    // Accumulate root node.
    //

    REQUIRE(rootNode);

    rootNode->totalTimeSum = 0;

    for (ProfilerNode* p = rootNode->lastChild; p != 0; p = p->prevBrother)
        rootNode->totalTimeSum += p->totalTimeSum;

    rootNode->counter = cycleCount;

    //
    // Accumulate subtree device time
    //

    profilerUpdateDeviceTreeTime(*rootNode);

    //
    // Print tree
    //

    REQUIRE(divTicksPerSec > 0);
    float32 tickFactor = divTicksPerSec;

    float32 fullRunTime = float32(rootNode->totalTimeSum) * tickFactor; 

    ////

    PrintNode visitor(processingThroughput, kit);
    DisplayReportParams o(tickFactor, cycleCount, fullRunTime, visitor);
    require(displayProfilerTree(*rootNode, o, 0, 0, stdPass));

    returnTrue;
}

//----------------------------------------------------------------

}
