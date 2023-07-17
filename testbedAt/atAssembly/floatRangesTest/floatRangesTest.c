#include "floatRangesTest.h"

#include "storage/classThunks.h"
#include "userOutput/printMsgEx.h"
#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"

namespace floatRangesTest {

//================================================================
//
// FloatRangesTestImpl
//
//================================================================

class FloatRangesTestImpl
{

public:

    void serialize(const ModuleSerializeKit& kit);
    stdbool process(stdPars(ProcessKit));

private:

    StandardSignal testSignal;

};

//================================================================
//
// FloatRangesTestImpl::serialize
//
//================================================================

void FloatRangesTestImpl::serialize(const ModuleSerializeKit& kit)
{
    testSignal.serialize(kit, STR("Float Ranges Test"), STR(""));
}

//================================================================
//
// convertThruFloat
//
//================================================================

template <typename Type>
float64 convertThruFloat(const Type& value)
{
    volatile float32 result = float32(value);
    return result;
}

//================================================================
//
// findFloatFloorCeil
//
//================================================================

template <typename Int>
stdbool findFloatFloorCeil(Int intValue, stdPars(DiagnosticKit))
{
    float64 exactValue = intValue;

    Int i = intValue;

    while_not (convertThruFloat(i) >= exactValue)
        ++i;

    while_not (convertThruFloat(i) <= exactValue)
        --i;

    printMsg(kit.msgLog, STR("floatDown(%) = %"), intValue, fltg(convertThruFloat(i), 10));

    while_not (convertThruFloat(i) >= exactValue)
        ++i;

    printMsg(kit.msgLog, STR("floatUp(%) = %"), intValue, fltg(convertThruFloat(i), 10));

    ////

    returnTrue;
}

//================================================================
//
// FloatRangesTestImpl::process
//
//================================================================

stdbool FloatRangesTestImpl::process(stdPars(ProcessKit))
{
    if_not (testSignal != 0)
        returnTrue;

    ////

    require(findFloatFloorCeil(0x7FFFFFFF, stdPass));

    require(findFloatFloorCeil(0xFFFFFFFF, stdPass));

    ////

    returnTrue;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(FloatRangesTest)
CLASSTHUNK_VOID1(FloatRangesTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_STD0(FloatRangesTest, process, ProcessKit)

//----------------------------------------------------------------

}
