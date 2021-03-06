#include "testShell.h"

#include "storage/classThunks.h"
#include "cfg/cfgInterface.h"
#include "compileTools/classContext.h"
#include "tests/resamplingTest/resamplingTest.h"
#include "tests/rotation3dTest/rotation3dTest.h"
#include "tests/gaussPresentationTest/gaussPresentationTest.h"
#include "tests/fourierFilterBank/fourierFilterBank.h"

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

    void setInputResolution(const Point<Space>& frameSize)
        {return base->setInputResolution(frameSize);}

    void serialize(const ModuleSerializeKit& kit);

    void inputMetadataSerialize(const InputMetadataSerializeKit& kit)
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

    fourierFilterBank::FourierFilterBank fourierFilterBank;
    gaussPresentationTest::GaussPresentationTest gaussPresentationTest;

    resamplingTest::ResamplingTest resamplingTest;
    Rotation3DTest rotation3dTest;

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
        CFG_NAMESPACE("~Tests");

        {
            CFG_NAMESPACE("Fourier Filter Bank");
            fourierFilterBank.serialize(kit);
        }

        {
            CFG_NAMESPACE("Gauss Presentation Test");
            gaussPresentationTest.serialize(kit);
        }

        {
            CFG_NAMESPACE("Resampling Test");
            resamplingTest.serialize(kit);
        }

        {
            CFG_NAMESPACE("Rotation 3D Test");
            rotation3dTest.serialize(kit);
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

    if (fourierFilterBank.active())
    {
        require(fourierFilterBank.process(0, stdPass));
        returnTrue;
    }

    if (gaussPresentationTest.active())
    {
        require(gaussPresentationTest.process(0, stdPass));
        returnTrue;
    }

    if (resamplingTest.active())
    {
        errorBlock(resamplingTest.process(resamplingTest::ProcessParams(kit.gpuRgbFrame), stdPass));
        returnTrue;
    }

    if (rotation3dTest.active())
    {
        errorBlock(rotation3dTest.process(stdPass));
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
