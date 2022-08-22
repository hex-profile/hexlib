#pragma once

#include "profilerShell/profiler/profilerStruct.h"
#include "kits/moduleHeader.h"
#include "userOutput/diagnosticKit.h"

namespace profilerReport {

//================================================================
//
// ReportKit
//
//================================================================

using ReportKit = DiagnosticKit;
using ReportFileKit = ReportKit;

//================================================================
//
// MakeReportParams
//
//================================================================

struct MakeReportParams
{
    ProfilerNode* rootNode;
    float32 divTicksPerSec;
    uint32 cycleCount;
    float32 processingThroughput;
    const CharType* outputDir;
};

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

    void serialize(const CfgSerializeKit& kit);
    stdbool makeReport(const MakeReportParams& o, stdPars(ReportFileKit));

private:

    DynamicClass<class HtmlReportImpl> instance;

};

//----------------------------------------------------------------

}
