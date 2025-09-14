#include "testShell.h"

#include "storage/classThunks.h"
#include "cfg/cfgInterface.h"
#include "compileTools/classContext.h"
#include "tests/resamplingTest/resamplingTest.h"
#include "tests/rotation3dTest/rotation3dTest.h"
#include "tests/gaussPresentationTest/gaussPresentationTest.h"
#include "tests/fourierFilterBank/fourierFilterBank.h"
#include "tests/mallocTest/mallocTest.h"
#include "tests/exceptionTest/exceptionTest.h"
#include "tests/formatTest/formatTest.h"
#include "tests/atanTest/atanTest.h"
#include "tests/quatGenTest/quatGenTest.h"
#include "tests/popCountTest/popCountTest.h"

namespace testShell {

//================================================================
//
// TestShellImpl
//
//================================================================

class TestShellImpl : public TestShell
{

public:

    void serialize(const ModuleSerializeKit& kit);
    stdbool process(Process& baseProcess, stdPars(ProcessKit));

private:

    fourierFilterBank::FourierFilterBank fourierFilterBank;
    gaussPresentationTest::GaussPresentationTest gaussPresentationTest;
    resamplingTest::ResamplingTest resamplingTest;
    Rotation3DTest rotation3dTest;
    UniqueInstance<MallocTest> mallocTest;
    UniqueInstance<AtanTest> atanTest;
    UniqueInstance<FormatTest> formatTest;
    UniqueInstance<QuatGenTest> quatGenTest;
    UniqueInstance<PopCountTest> popCountTest;
    UniqueInstance<ExceptionTest> exceptionTest;

};

UniquePtr<TestShell> TestShell::create() {return makeUnique<TestShellImpl>();}

//================================================================
//
// TestShellImpl::serialize
//
//================================================================

void TestShellImpl::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Tests");

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

        {
            CFG_NAMESPACE("Malloc Test");
            mallocTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("Atan Test");
            atanTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("Format Test");
            formatTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("QuatGen Test");
            quatGenTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("Pop Count Test");
            popCountTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("Exception Test");
            exceptionTest->serialize(kit);
        }
    }
}

//================================================================
//
// TestShellImpl::process
//
//================================================================

stdbool TestShellImpl::process(Process& baseProcess, stdPars(ProcessKit))
{

    //----------------------------------------------------------------
    //
    // Tests.
    //
    //----------------------------------------------------------------

    if (fourierFilterBank.active())
    {
        require(fourierFilterBank.process({}, stdPass));
        returnTrue;
    }

    if (gaussPresentationTest.active())
    {
        require(gaussPresentationTest.process({}, stdPass));
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

    if (mallocTest->active())
    {
        require(mallocTest->process(stdPass));
        returnTrue;
    }

    if (atanTest->active())
    {
        require(atanTest->process(stdPass));
        returnTrue;
    }

    if (formatTest->active())
    {
        require(formatTest->process(stdPass));
        returnTrue;
    }

    if (quatGenTest->active())
    {
        require(quatGenTest->process(stdPass));
        returnTrue;
    }

    if (popCountTest->active())
    {
        require(popCountTest->process(stdPass));
        returnTrue;
    }

    if (exceptionTest->active())
    {
        require(exceptionTest->process(stdPass));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    require(baseProcess(stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
