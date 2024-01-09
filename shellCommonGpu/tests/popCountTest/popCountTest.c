#include "popCountTest.h"

#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"
#include "gpuDevice/gpuDeviceEmu.h"
#include "rndgen/rndgenFloat.h"
#include "userOutput/printMsgEx.h"

namespace popCountTest {

//================================================================
//
// simplePopCount
//
//================================================================

int simplePopCount(uint32 n)
{
    int count = 0;

    for_count (i, 32)
    {
        if (n & 1)
            ++count;

        n >>= 1;
    }

    return count;
}

//================================================================
//
// PopCountTestImpl
//
//================================================================

class PopCountTestImpl : public PopCountTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, PopCountTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0x92709DD2u> displaySwitch;

    RndgenState rndgen = 0x9A4DB5F1u;

};

//----------------------------------------------------------------

UniquePtr<PopCountTest> PopCountTest::create()
{
    return makeUnique<PopCountTestImpl>();
}

//================================================================
//
// PopCountTestImpl::serialize
//
//================================================================

void PopCountTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize(kit, STR("Display"), {STR("<Nothing>")}, {STR("Pop Count Test"), STR("Ctrl+Alt+Shift+P")});
}

//================================================================
//
// PopCountTestImpl::process
//
//================================================================

stdbool PopCountTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    ////

    for_count (i, 100000)
    {
        uint32 n = rndgenUniform<uint32>(rndgen);
        int simpleCount = simplePopCount(n);
        int devCount = devPopCount(n);

        REQUIRE(simpleCount == devCount);
    }

    printMsgL(kit, STR("PopCount test OK"));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
