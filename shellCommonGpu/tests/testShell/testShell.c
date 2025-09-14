#include "testShell.h"

#include "cfg/cfgInterface.h"
#include "compileTools/classContext.h"
#include "storage/classThunks.h"
#include "tests/atanTest/atanTest.h"
#include "tests/exceptionTest/exceptionTest.h"
#include "tests/fontTest/fontTest.h"
#include "tests/formatTest/formatTest.h"
#include "tests/fourierFilterBank/fourierFilterBank.h"
#include "tests/gaussPresentationTest/gaussPresentationTest.h"
#include "tests/mallocTest/mallocTest.h"
#include "tests/popCountTest/popCountTest.h"
#include "tests/preciseIntrinsicsCalib/picPatternTest.h"
#include "tests/quatGenTest/quatGenTest.h"
#include "tests/resamplingTest/resamplingTest.h"
#include "tests/rotation3dTest/rotation3dTest.h"

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
    void process(Process& baseProcess, stdPars(ProcessKit));

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
    UniqueInstance<FontTest> fontTest;
    UniqueInstance<PicPatternTest> picPatternTest;

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

        {
            CFG_NAMESPACE("Font Test");
            fontTest->serialize(kit);
        }

        {
            CFG_NAMESPACE("Precise Intrinsics Calibration");

            {
                CFG_NAMESPACE("Pattern Generator");
                picPatternTest->serialize(kit);
            }
        }
    }
}

//================================================================
//
// TestShellImpl::process
//
//================================================================

void TestShellImpl::process(Process& baseProcess, stdPars(ProcessKit))
{

    //----------------------------------------------------------------
    //
    // Tests.
    //
    //----------------------------------------------------------------

    if (fourierFilterBank.active())
    {
        fourierFilterBank.process({}, stdPass);
        return;
    }

    if (gaussPresentationTest.active())
    {
        gaussPresentationTest.process({}, stdPass);
        return;
    }

    if (resamplingTest.active())
    {
        errorBlock(resamplingTest.process(resamplingTest::ProcessParams(kit.gpuRgbFrame), stdPassNc));
        return;
    }

    if (rotation3dTest.active())
    {
        errorBlock(rotation3dTest.process(stdPassNc));
        return;
    }

    if (mallocTest->active())
    {
        mallocTest->process(stdPass);
        return;
    }

    if (atanTest->active())
    {
        atanTest->process(stdPass);
        return;
    }

    if (formatTest->active())
    {
        formatTest->process(stdPass);
        return;
    }

    if (quatGenTest->active())
    {
        quatGenTest->process(stdPass);
        return;
    }

    if (popCountTest->active())
    {
        popCountTest->process(stdPass);
        return;
    }

    if (exceptionTest->active())
    {
        exceptionTest->process(stdPass);
        return;
    }

    if (fontTest->active())
    {
        fontTest->process(stdPass);
        return;
    }

    if (picPatternTest->active())
    {
        picPatternTest->process(stdPass);
        return;
    }

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    baseProcess(stdPass);
}

//----------------------------------------------------------------

}
