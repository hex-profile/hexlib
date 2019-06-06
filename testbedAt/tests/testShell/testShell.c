#include "testShell.h"

#include "storage/classThunks.h"
#include "tests/resampleTest/resampleTest.h"
#include "cfg/cfgInterface.h"
#include "compileTools/classContext.h"

namespace testShell {

//================================================================
//
// TestShellImpl
//
//================================================================

class TestShellImpl : public AtEngine
{

public:

    TestShellImpl(UniquePtr<AtEngine> base)
        : base(move(base)) {}

public:

    CharType* getName() const
        {return base->getName();}

    void serialize(const ModuleSerializeKit& kit);

    void setInputResolution(const Point<Space>& frameSize)
        {return base->setInputResolution(frameSize);}

    void inputMetadataReset()
        {return base->inputMetadataReset();}

    void inputMetadataSerialize(const CfgSerializeKit& kit)
        {return base->inputMetadataSerialize(kit);}

    bool reallocValid() const
        {return base->reallocValid();}

    stdbool realloc(stdPars(AtEngineReallocKit))
        {return base->realloc(stdPassThru);}

    void inspectProcess(ProcessInspector& inspector)
        {return base->inspectProcess(inspector);}

    stdbool process(stdPars(AtEngineProcessKit));

private:

    UniquePtr<AtEngine> base;

private:

    resampleTest::ResampleTest resampleTest;

};

//----------------------------------------------------------------

UniquePtr<AtEngine> testShellCreate(UniquePtr<AtEngine> base)
    {return makeUnique<TestShellImpl>(move(base));}

//================================================================
//
// TestShellImpl::serialize
//
//================================================================

void TestShellImpl::serialize(const ModuleSerializeKit& kit)
{
    base->serialize(kit);

    ////

    {
        CFG_NAMESPACE("Tests");

        {
            CFG_NAMESPACE("Resampling Test");
            resampleTest.serialize(kit);
        }
    }
}

//================================================================
//
// TestShellImpl::process
//
//================================================================

stdbool TestShellImpl::process(stdPars(AtEngineProcessKit))
{

    //----------------------------------------------------------------
    //
    // Tests.
    //
    //----------------------------------------------------------------

    if (resampleTest.active())
    {
        require(resampleTest.process(resampleTest::ProcessParams(kit.gpuRgbFrame), stdPass));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    require(base->process(stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
