#include "quatGenTest.h"

#include "cfgTools/multiSwitch.h"
#include "dataAlloc/arrayMemory.h"
#include "errorLog/errorLog.h"
#include "mathFuncs/rotationMath.h"
#include "mathFuncs/rotationMath3d.h"
#include "numbers/getBits.h"
#include "rndgen/randRange.h"
#include "rndgen/rndgenFloat.h"
#include "rndgen/rndgenPoint.h"
#include "stl/stlArray.h"
#include "userOutput/printMsgEx.h"
#include "vectorTypes/vectorType.h"

namespace quatGenTest {

//================================================================
//
// Float
//
//================================================================

using Float = float32;

//================================================================
//
// randomUnitVector
//
//================================================================

sysinline auto randomUnitVector(RndgenState& rndgen)
{
    auto r = rndgenGaussApproxFour<Point3D<float32>>(rndgen);
    auto result = vectorNormalize(convertNearest<Float>(r));
    return result;
}

//================================================================
//
// calculateError
//
//================================================================

sysinline auto calculateError(const Point3D<Float>& v, const Point3D<Float>& t)
{
    return vectorLength(v - t);
}

//================================================================
//
// QuatGenTestImpl
//
//================================================================

class QuatGenTestImpl : public QuatGenTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, QuatGenTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0x6E58C360> displaySwitch;

    RndgenState rndgen = 0x6A1264FBu;
    Float worstError = 0;
    Point3D<Float> worstV = point3D<Float>(0);
    Point3D<Float> worstT = point3D<Float>(0);

};

//----------------------------------------------------------------

UniquePtr<QuatGenTest> QuatGenTest::create()
{
    return makeUnique<QuatGenTestImpl>();
}

//================================================================
//
// QuatGenTestImpl::serialize
//
//================================================================

void QuatGenTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize(kit, STR("Display"), {STR("<Nothing>")}, {STR("QuatGen Test"), STR("Ctrl+Alt+Q")});
}

//================================================================
//
// QuatGenTestImpl::process
//
//================================================================

stdbool QuatGenTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    ////

    printMsgL(kit, STR("QuatGen test, worst precision % bits"), fltf(getBits(worstError, 64), 1));
    printMsgL(kit, STR("V = %"), fltf(worstV, 6));
    printMsgL(kit, STR("T = %"), fltf(worstT, 6));

    for_count (i, 10000)
    {
        auto scale = randRange<Float>(rndgen, 0.8f, 1.f);

        auto V = scale * randomUnitVector(rndgen);
        auto T = scale * randomUnitVector(rndgen);

        if (randRange(rndgen, 0, 63) == 0)
        {
            V = T;

            if (randRange(rndgen, 0, 63) == 1)
                V += 1e-18f * randomUnitVector(rndgen);
        }

        REQUIRE(def(V, T));

        auto rotation = computeRotationQuaternion(V, T);
        auto rotatedV = rotation % V;

        auto error = calculateError(rotatedV, T);
        REQUIRE(def(error));

        auto oldError = worstError;

        if (error > worstError)
        {
            worstError = error;
            worstT = T;
            worstV = V;
        }
    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
