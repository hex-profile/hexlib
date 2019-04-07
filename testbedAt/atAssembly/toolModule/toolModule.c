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
        CFG_NAMESPACE_MODULE("Video Preprocessor");
        videoPreprocessor.serialize(kit);
    }

    {
        CFG_NAMESPACE_MODULE("Gpu Overhead Test");
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

bool ToolModule::realloc(stdPars(ReallocKit))
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

bool ToolModule::process(ToolTarget& toolTarget, stdPars(ProcessKit))
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
