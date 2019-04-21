#pragma once

#include "profilerShell/profiler/profilerStruct.h"
#include "interfaces/fileTools.h"
#include "kits/moduleHeader.h"

namespace profilerReport {

//================================================================
//
// ReportKit
//
//================================================================

KIT_COMBINE2(ReportKit, ErrorLogKit, MsgLogKit);
KIT_COMBINE2(ReportFileKit, ReportKit, FileToolsKit);

//================================================================
//
// MakeReportParams
//
//================================================================

KIT_CREATE5
(
    MakeReportParams,
    ProfilerNode*, rootNode,
    float32, divTicksPerSec,
    uint32, cycleCount,
    float32, processingThroughput,
    const CharType*, outputDir
);

//================================================================
//
// HtmlReport
//
//================================================================

class HtmlReport
{

public:

    HtmlReport();
    ~HtmlReport();

public:

    void serialize(const ModuleSerializeKit& kit);
    stdbool makeReport(const MakeReportParams& o, stdPars(ReportFileKit));

private:

    StaticClass<class HtmlReportImpl, 1 << 8> instance;

};

//----------------------------------------------------------------

}
