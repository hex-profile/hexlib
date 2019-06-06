#include "toolModule.h"

#include "storage/classThunks.h"

namespace toolModule {

//================================================================
//
// ToolModuleImpl
//
//================================================================

class ToolModuleImpl
{

public:

    void serialize(const ModuleSerializeKit& kit);

    void setFrameSize(const Point<Space>& frameSize)
        {videoPreprocessor.setFrameSize(frameSize);}

    bool reallocValid() const
        {return videoPreprocessor.reallocValid();}

    stdbool realloc(stdPars(ReallocKit))
        {return videoPreprocessor.realloc(stdPass);}

    Point<Space> outputFrameSize() const
        {return videoPreprocessor.outputFrameSize();}

    stdbool process(ToolTarget& target, stdPars(ProcessKit));

private:

    videoPreprocessor::VideoPreprocessor videoPreprocessor;

};

//----------------------------------------------------------------

CLASSTHUNK_CONSTRUCT_DESTRUCT(ToolModule)
CLASSTHUNK_VOID1(ToolModule, serialize, const ModuleSerializeKit&)
CLASSTHUNK_VOID1(ToolModule, setFrameSize, const Point<Space>&)
CLASSTHUNK_BOOL_CONST0(ToolModule, reallocValid)
CLASSTHUNK_BOOL_STD0(ToolModule, realloc, ReallocKit)
CLASSTHUNK_PURE0(ToolModule, Point<Space>, point(0), outputFrameSize, const)
CLASSTHUNK_BOOL_STD1(ToolModule, process, ToolTarget&, ProcessKit)

//================================================================
//
// ToolModuleImpl::serialize
//
//================================================================

void ToolModuleImpl::serialize(const ModuleSerializeKit& kit)
{
    videoPreprocessor.serialize(kit);
}

//================================================================
//
// ToolModuleImpl::process
//
//================================================================

stdbool ToolModuleImpl::process(ToolTarget& target, stdPars(ProcessKit))
{

    //----------------------------------------------------------------
    //
    // Process
    //
    //----------------------------------------------------------------

    require(videoPreprocessor.process(target, stdPass));

    returnTrue;
}

//----------------------------------------------------------------

}
