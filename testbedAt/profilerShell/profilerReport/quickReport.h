#pragma once

#include "profilerShell/profiler/profilerStruct.h"
#include "kits/moduleHeader.h"

namespace profilerQuickReport {

//================================================================
//
// ReportKit
//
//================================================================

KIT_COMBINE2(ReportKit, ErrorLogKit, MsgLogKit);

//================================================================
//
// namedNodesReport
//
//================================================================

bool namedNodesReport
(
    ProfilerNode* rootNode,
    float32 divTicksPerSec,
    uint32 cycleCount,
    float32 processingThroughput,
    stdPars(ReportKit)
);

//----------------------------------------------------------------

}
