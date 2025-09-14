#include "atanTest.h"

#include "gpuSupport/gpuTool.h"
#include "mathFuncs/rotationMath.h"

#if HOSTCODE
#include "cfgTools/boolSwitch.h"
#include "dataAlloc/arrayMemory.h"
#include "numbers/getBits.h"
#include "rndgen/rndgenFloat.h"
#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgEx.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "dataAlloc/gpuArrayMemory.h"
#endif

namespace atanTest {

//================================================================
//
// computePhase
//
//================================================================

GPUTOOL_1D_BEG
(
    computePhase,
    PREP_EMPTY,
    ((const Point<float32>, src))
    ((float32, dst)),
    ((bool, testApproxPhase))
)
#if DEVCODE
{
    auto value = helpRead(*src);
    *dst = testApproxPhase ? approxPhase(value) : fastPhase(value);
}
#endif
GPUTOOL_1D_END

//================================================================
//
// AtanTestImpl
//
//================================================================

#if HOSTCODE

class AtanTestImpl : public AtanTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    void process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, AtanTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0xE26D8384> displaySwitch;

    NumericVar<int32> testCount{1, typeMax<int32>(), 65536};
    BoolSwitch testApproxPhase{false};

    RndgenState rndgen = 1;
    float32 maxError = 0;

};

//----------------------------------------------------------------

UniquePtr<AtanTest> AtanTest::create()
{
    return makeUnique<AtanTestImpl>();
}

//================================================================
//
// AtanTestImpl::serialize
//
//================================================================

void AtanTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize(kit, STR("Display"), {STR("<Nothing>"), STR("")}, {STR("Atan Test")});
    testCount.serialize(kit, STR("Test Count"));
    testApproxPhase.serialize(kit, STR("Test Approx Phase"));
}

//================================================================
//
// AtanTestImpl::process
//
//================================================================

void AtanTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        return;

    GpuCopyThunk gpuCopy;

    //----------------------------------------------------------------
    //
    // Generate.
    //
    //----------------------------------------------------------------

    Space size = testCount();

    ARRAY_ALLOC_FOR_GPU_EXCH(srcCpu, Point<float32>, size);

    if (kit.dataProcessing)
    {
        for_count (i, testCount())
        {
            auto r = rndgenUniform<float32>(rndgen);
            auto p = circleCCW(r);
            p *= rndgenLogScale(rndgen, 1.f/16, 16.f);
            srcCpu[i] = p;
        }
    }

    ////

    GPU_ARRAY_ALLOC(srcGpu, Point<float32>, size);

    gpuCopy(srcCpu, srcGpu, stdPass);

    //----------------------------------------------------------------
    //
    // Process and copy back.
    //
    //----------------------------------------------------------------

    GPU_ARRAY_ALLOC(testGpu, float32, size);

    computePhase(srcGpu, testGpu, testApproxPhase, stdPass);

    ////

    ARRAY_ALLOC_FOR_GPU_EXCH(testCpu, float32, size);

    gpuCopy(testGpu, testCpu, stdPass);

    gpuCopy.waitClear();

    //----------------------------------------------------------------
    //
    // Normalize phase.
    //
    //----------------------------------------------------------------

    auto normalizePhase = [] (float32 value)
    {
        while (value < 0) value += 1;
        while (value >= 1) value -= 1;
        return value;
    };

    //----------------------------------------------------------------
    //
    // Verify.
    //
    //----------------------------------------------------------------

    if (kit.dataProcessing)
    {
        for_count (i, size)
        {
            auto p = srcCpu[i];

            auto ref = normalizePhase(float32((0.5 / pi64) * atan2(float64{p.Y}, float64{p.X})));
            auto opt = normalizePhase(testCpu[i]);

            REQUIRE(ref >= 0 && ref < 1);
            REQUIRE(opt >= 0 && opt < 1);
            auto distance = circularDistance(ref, opt);

            maxError = maxv(maxError, distance);
        }
    }

    ////

    printMsgL(kit, STR("Atan test: Accuracy % bits"), fltf(getBits(maxError), 2));
}

#endif

//----------------------------------------------------------------

}
