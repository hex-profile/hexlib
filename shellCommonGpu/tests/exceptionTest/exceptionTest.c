#include "exceptionTest.h"

#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgEx.h"
#include "tests/exceptionTest/exceptionTestAux.h"
#include "compileTools/blockExceptionsSilent.h"
#include "timer/timer.h"
#include "formatting/prettyNumber.h"
#include "userOutput/diagnosticKit.h"

namespace exceptionTest {

//================================================================
//
// NonTrivialClass
//
//================================================================

class NonTrivialClass
{

public:

    sysinline NonTrivialClass()
        {memory = myMalloc(1);}

    sysinline ~NonTrivialClass()
        {myFree(memory);}

private:

    void* memory = nullptr;

};

//================================================================
//
// CLEANUP_COUNT
//
//================================================================

#define CLEANUP_COUNT 101

//================================================================
//
// RETCODE_CHECK
//
//================================================================

#define RETCODE_CHECK(condition) \
    if (!(condition)) return false

#define RETCODE_TEST(number) \
    NonTrivialClass local##number; \
    RETCODE_CHECK(value != number)

//================================================================
//
// returnCodeInnerFunc
//
//================================================================

sysnoinline bool returnCodeInnerFunc(int value)
{
    #define TMP_MACRO(i, _) \
        RETCODE_TEST(i);

    PREP_FOR(CLEANUP_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    RETCODE_TEST(1013);
    return true;
}

//================================================================
//
// returnCodeTest
//
//================================================================

sysnoinline bool returnCodeTest(int value)
{
    RETCODE_CHECK(returnCodeInnerFunc(value));

    return true;
}

//================================================================
//
// EXCEPTION_CHECK
//
//================================================================

#define EXCEPTION_CHECK(condition) \
    if (condition) ; else throwFailure()

#define EXCEPTION_TEST(number) \
    NonTrivialClass local##number; \
    EXCEPTION_CHECK(value != number)

//================================================================
//
// exceptionCodeInnerFunc
//
//================================================================

sysnoinline void exceptionCodeInnerFunc(int value)
{
    #define TMP_MACRO(i, _) \
        EXCEPTION_TEST(i);

    PREP_FOR(CLEANUP_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    EXCEPTION_TEST(1013);
}

//================================================================
//
// exceptionCodeTest
//
//================================================================

sysnoinline void exceptionCodeTest(int value)
{
    exceptionCodeInnerFunc(value);
}

//================================================================
//
// newApproachCaller
//
//================================================================

#define stdCall \
    stdPass); do {if (!successFlag) return;} while (0

//----------------------------------------------------------------

sysnoinline void newApproachCaller(bool& successFlag, int value, stdPars(DiagnosticKit))
{
    if (value == 0)
        newApproachCallee(successFlag, stdCall);

    newApproachCallee(successFlag, stdCall);
    newApproachCallee(successFlag, stdCall);
    newApproachCallee(successFlag, stdCall);
}

//================================================================
//
// ExceptionTestImpl
//
//================================================================

class ExceptionTestImpl : public ExceptionTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, ExceptionTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0xC1A83639u> displaySwitch;

    int32 retcodeSuccessChunkSize = 1;
    int32 retcodeFailChunkSize = 1;
    int32 exceptionSuccessChunkSize = 1;
    int32 exceptionFailChunkSize = 1;

};

//----------------------------------------------------------------

UniquePtr<ExceptionTest> ExceptionTest::create()
{
    return makeUnique<ExceptionTestImpl>();
}

//================================================================
//
// ExceptionTestImpl::serialize
//
//================================================================

void ExceptionTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit,
        STR("Display"),
        {STR("<Nothing>")},
        {STR("Exceptions Test"), STR("Ctrl+Alt+Shift+E")}
    );
}

//================================================================
//
// ExceptionTestImpl::process
//
//================================================================

stdbool ExceptionTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    if_not (kit.dataProcessing)
        returnTrue;

    ////

    printMsgL(kit, STR("Exception test"));

    ////

    bool successFlag = true;

    newApproachCaller(successFlag, 0, stdPass);

    ////

    int valueSuccess = 1000;
    int valueFail = 1013;

    //----------------------------------------------------------------
    //
    // genericTest
    //
    //----------------------------------------------------------------

    auto genericTest = [] (auto name, auto iteration, int32& chunkSize, stdPars(auto))
    {
        auto startMoment = kit.timer.moment();
        auto currentMoment = startMoment;

        ////

        int32 chunkCount = 0;

        for (;;)
        {
            for_count (c, chunkSize)
                iteration();

            ++chunkCount;

            currentMoment = kit.timer.moment();

            if (kit.timer.diff(startMoment, currentMoment) >= 1.0f)
                break;
        }

        ////

        auto passedTime = kit.timer.diff(startMoment, currentMoment);
        auto iterationTime = passedTime / chunkCount / chunkSize;

        printMsgL(kit, STR("%: Time %s (chunk count %, chunk size %, test time %s)"),
            name, prettyNumber(iterationTime), prettyNumber(chunkCount), prettyNumber(chunkSize), prettyNumber(passedTime));

        ////

        auto correctionFactor = chunkCount / 4.f;
        chunkSize = clampMin(convertNearest<int32>(chunkSize * correctionFactor), 1);

        returnTrue;
    };

    //----------------------------------------------------------------
    //
    // Test return mode.
    //
    //----------------------------------------------------------------

    auto retcodeIterationSuccess = [&] ()
        {returnCodeTest(valueSuccess);};

    auto retcodeIterationFail = [&] ()
        {returnCodeTest(valueFail);};

    require(genericTest(STR("Retcode Success"), retcodeIterationSuccess, retcodeSuccessChunkSize, stdPass));
    require(genericTest(STR("Retcode Fail"), retcodeIterationFail, retcodeFailChunkSize, stdPass));

    //----------------------------------------------------------------
    //
    // Test exception mode.
    //
    //----------------------------------------------------------------

    auto exceptionIterationSuccess = [&] ()
    {
        exceptionCodeTest(valueSuccess);
    };

    auto exceptionIterationFail = [&] ()
    {
        try {exceptionCodeTest(valueFail);}
        catch (...) {}
    };

    require(genericTest(STR("Except Success"), exceptionIterationSuccess, exceptionSuccessChunkSize, stdPass));
    require(genericTest(STR("Except Fail"), exceptionIterationFail, exceptionFailChunkSize, stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
