#include "toolModule.h"

#include "cfg/cfgInterface.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// ToolModule::serialize
//
//================================================================

void ToolModule::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Video Preprocessor");
        videoPreprocessor.serialize(kit);
    }

    {
        CFG_NAMESPACE("Gpu Overhead Test");
        gpuOverheadTest.serialize(kit);
    }

}

//================================================================
//
// ToolModule::reallocValid
//
//================================================================

bool ToolModule::reallocValid() const 
{
    return 
        allv(allocFrameSize == frameSize) &&
        videoPreprocessor.reallocValid();
}

//================================================================
//
// ToolModule::realloc
//
//================================================================

stdbool ToolModule::realloc(stdPars(ReallocKit))
{
    stdBegin;
    allocFrameSize = point(0);
    require(videoPreprocessor.realloc(frameSize, stdPass));
    allocFrameSize = frameSize;
    stdEnd;
}

//================================================================
//
// ToolModule::process
//
//================================================================

stdbool ToolModule::process(ToolTarget& toolTarget, stdPars(ProcessKit))
{
    stdBegin;

    //
    // Overhead test
    //

    require(gpuOverheadTest.process(stdPass));

    //
    // Continue with video processor shell
    //

    require(videoPreprocessor.processEntry(toolTarget, stdPass));

    stdEnd;
}
