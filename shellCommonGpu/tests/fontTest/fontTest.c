#include "fontTest.h"

#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgEx.h"
#include "tests/fontTest/fontTypes.h"

namespace fontTest {

using namespace fontTypes;

//================================================================
//
// FontTestImpl
//
//================================================================

class FontTestImpl : public FontTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, FontTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0xBB5A342Bu> displaySwitch;

};

////

UniquePtr<FontTest> FontTest::create()
    {return makeUnique<FontTestImpl>();}

//================================================================
//
// FontTestImpl::serialize
//
//================================================================

void FontTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit,
        STR("Display"),
        {STR("<Nothing>")},
        {STR("Font Test"), STR("Ctrl+Alt+Shift+T")}
    );
}

//================================================================
//
// FontTestImpl::process
//
//================================================================

stdbool FontTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    ////

    printMsgL(kit, STR("Font test"));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
