#include <omp.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "numbers/float/floatType.h"

//================================================================
//
// breakFlow
//
//================================================================

void breakFlowDefault() {}
void (*breakFlow)() = breakFlowDefault;

//================================================================
//
// Test conversion float -> int
//
//================================================================

template <typename Int, Rounding rounding>
void floatIntConvTest(CharArray name, MsgLog& msgLog)
{

#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#endif

    ////

    volatile bool globOk = true;
    uint32 globBadValue = 0;

    ////

    pragmaOmp(parallel for)

    for (int32 hi = 0; hi <= 0xFFFF; ++hi)
    {
        if (globOk)
        {
            bool locOk = true;
            uint32 locBadValue = 0;

            for (int32 lo = 0; lo <= 0xFFFF; ++lo)
            {
                if_not (globOk)
                    break;

                uint32 intVal = (hi << 16) | lo;

                volatile float32 testValData = 0;
                * ((uint32*) &testValData) = intVal;

                volatile float32 testVal = testValData;

                ////

                volatile float64 testVald = testVal;


                Int ref =
                    rounding == RoundDown ? Int(floorv(testVald)) :
                    rounding == RoundUp ? Int(ceilv(testVald)) :
                    rounding == RoundNearest ? Int(floorv(testVald + 0.5)) :
                    0;

                bool refOk = def(float32(testVal)) && (absv(ref - testVald) <= 1);

                //////

                Int test = 0;
                bool convertOk = convertFlex<rounding, ConvertNormal>(float32(testVal), test);

                ////

                check_flag(refOk == convertOk, locOk);

                if (convertOk)
                {
                    if_not (test == ref)
                    {
                        if (rounding == RoundNearest)
                            check_flag(floorv(testVald) + 0.5f == testVald, locOk);
                        else
                            locOk = false;
                    }
                }

                if_not (locOk)
                {
                    locBadValue = intVal;
                    break;
                }

            }

            if_not (locOk)
            {
                pragmaOmp(critical)
                {
                    if (globOk)
                    {
                        globOk = false;
                        globBadValue = locBadValue;
                    }
                }
            }

        }
    }

    ////

    printMsg
    (
        msgLog, STR("%0/%1: %2 (%3)"),
        name,
        rounding,
        globOk ? STR("OK") : STR("ERROR!"),
        hex(globBadValue),
        globOk ? msgInfo : msgWarn
    );

    msgLog.update();

}

//================================================================
//
// floatIntConvTestAllRoundModes
//
//================================================================

template <typename Int>
void floatIntConvTestAllRoundModes(CharArray name, MsgLog& msgLog)
{
    floatIntConvTest<Int, RoundDown>(name, msgLog);
    floatIntConvTest<Int, RoundUp>(name, msgLog);
    floatIntConvTest<Int, RoundNearest>(name, msgLog);
}

//================================================================
//
// floatIntConvTestAllTypes
//
//================================================================

void floatIntConvTestAllTypes(MsgLog& msgLog)
{
    msgLog.clear();
    msgLog.update();

    floatIntConvTestAllRoundModes<int32>(STR("int32"), msgLog);
    floatIntConvTestAllRoundModes<uint32>(STR("uint32"), msgLog);

    floatIntConvTestAllRoundModes<int16>(STR("int16"), msgLog);
    floatIntConvTestAllRoundModes<uint16>(STR("uint16"), msgLog);

    floatIntConvTestAllRoundModes<int8>(STR("int8"), msgLog);
    floatIntConvTestAllRoundModes<uint8>(STR("uint8"), msgLog);

    printMsg(msgLog, STR("Testing finished"));
}
