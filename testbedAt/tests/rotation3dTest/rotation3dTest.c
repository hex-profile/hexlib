#include "rotation3dTest.h"

#include "cfgTools/multiSwitch.h"
#include "storage/classThunks.h"
#include "userOutput/printMsgEx.h"

//================================================================
//
// Rotation3DTestImpl
//
//================================================================

class Rotation3DTestImpl
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != DisplayNothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum DisplayType {DisplayNothing, DisplayRotationTest, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0xAA589EE5> displaySwitch;

};

//----------------------------------------------------------------

CLASSTHUNK_CONSTRUCT_DESTRUCT(Rotation3DTest)
CLASSTHUNK_VOID1(Rotation3DTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(Rotation3DTest, active)
CLASSTHUNK_BOOL_STD0(Rotation3DTest, process, GpuModuleProcessKit)

//================================================================
//
// Rotation3DTestImpl::serialize
//
//================================================================

void Rotation3DTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit, STR("Display"), 
        {STR("<Nothing>"), STR("")},
        {STR("Rotation 3D Test"), STR("Alt+Shift+R")}
    );
}

//================================================================
//
// Rotation3DTestImpl::process
//
//================================================================

stdbool Rotation3DTestImpl::process(stdPars(GpuModuleProcessKit))
{
    stdBegin;

    DisplayType displayType = kit.outputLevel >= OUTPUT_RENDER ? displaySwitch : DisplayNothing;

    if (displayType == DisplayNothing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Test
    //
    //----------------------------------------------------------------

    printMsgL(kit, STR("Rotation 3D Math Test:"));

    ////

    stdEnd;
}
