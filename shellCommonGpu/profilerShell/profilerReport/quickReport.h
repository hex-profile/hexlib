#pragma once

#include "profilerShell/profiler/profilerStruct.h"
#include "kits/moduleHeader.h"

namespace profilerQuickReport {

//================================================================
//
// ReportKit
//
//================================================================

using ReportKit = KitCombine<ErrorLogKit, MsgLogKit>;

//================================================================
//
// namedNodesReport
//
//================================================================

void namedNodesReport
(
    ProfilerNode* rootNode,
    float32 divTicksPerSec,
    uint32 cycleCount,
    float32 processingThroughput,
    stdPars(ReportKit)
);

//----------------------------------------------------------------

}
