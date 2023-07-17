#include "formatTest.h"

#include "cfgTools/multiSwitch.h"
#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "formatting/messageFormatter.h"
#include "rndgen/randRange.h"
#include "simpleString/simpleString.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsgEx.h"

namespace formatTest {

//================================================================
//
// referenceFixedFormat
//
//================================================================

template <typename Uint, typename Writer>
stdbool referenceFixedFormat(Uint body, int expo, int fracDigits, const Writer& writer, stdPars(ErrorLogKit))
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Uint) && !TYPE_IS_SIGNED(Uint));
    REQUIRE(fracDigits >= 0);

    //----------------------------------------------------------------
    //
    // Utility.
    //
    //----------------------------------------------------------------

    auto writeZeros = [&] (int n)
    {
        for_count (i, n)
            writer("0", 1);
    };

    ////

    auto getFactor = [&] (int digits, Uint& factor) -> bool
    {
        factor = 1;

        for_count (i, digits)
        {
            constexpr auto maxVal = TYPE_MAX(Uint) / 10;
            ensure(factor <= maxVal);
            factor *= 10;
        }

        return true;
    };

    //----------------------------------------------------------------
    //
    // Reduce body digits with rounding.
    //
    //----------------------------------------------------------------

    {
        int shrAmount = clampMin(-expo, 0);
        auto cutAmount = clampMin(shrAmount - fracDigits, 0);

        if (cutAmount)
        {
            Uint cutFactor{};
            REQUIRE(getFactor(cutAmount, cutFactor));

            body = (body + (cutFactor / 2)) / cutFactor;
            expo += cutAmount;
        }
    }

    ////

    int shrAmount = clampMin(-expo, 0);
    REQUIRE(shrAmount >= 0 && shrAmount <= fracDigits);

    ////

    if (body == 0)
        expo = 0;

    //----------------------------------------------------------------
    //
    // Body.
    //
    //----------------------------------------------------------------

    COMPILE_ASSERT(TYPE_BIT_COUNT(Uint) <= 64);
    constexpr int maxUintDigits = 20;

    char bodyMemory[maxUintDigits];
    auto* bodyEnd = bodyMemory + COMPILE_ARRAY_SIZE(bodyMemory);
    auto* bodyPtr = bodyEnd;

    {
        auto value = body;

        do
        {
            auto valueDiv10 = value / 10;
            auto digit = value - valueDiv10 * 10;
            REQUIRE(bodyPtr != bodyMemory);
            *--bodyPtr = '0' + char(digit);
            value = valueDiv10;
        }
        while (value != 0);
    }

    //----------------------------------------------------------------
    //
    // Int part.
    //
    //----------------------------------------------------------------

    {
        int bodySize = bodyEnd - bodyPtr;
        int intSize = clampMin(bodySize - shrAmount, 0);

        writer(bodyPtr, intSize);
        bodyPtr += intSize;

        ////

        int shlAmount = clampMin(expo, 0);
        writeZeros(shlAmount);

        ////

        if (intSize == 0)
            writeZeros(1);
    }

    //----------------------------------------------------------------
    //
    // Frac part.
    //
    //----------------------------------------------------------------

    if (fracDigits != 0)
    {
        writer(".", 1);

        int capacity = fracDigits;

        ////

        int bodySize = bodyEnd - bodyPtr;

        {
            int n = clampMin(shrAmount - bodySize, 0);
            REQUIRE(n <= capacity);
            writeZeros(n);
            capacity -= n;
        }

        ////

        REQUIRE(bodySize <= capacity);

        ////

        writer(bodyPtr, bodySize);
        capacity -= bodySize;

        ////

        writeZeros(capacity);
    }

    ////

    returnTrue;
}

//================================================================
//
// FormatTestImpl
//
//================================================================

class FormatTestImpl : public FormatTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, FormatTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0x08CD0801> displaySwitch;

    RndgenState rndgen = 1;
    NumericVar<Space> iterationCount{1, typeMax<Space>(), 10000};

};

//----------------------------------------------------------------

UniquePtr<FormatTest> FormatTest::create()
{
    return makeUnique<FormatTestImpl>();
}

//================================================================
//
// FormatTestImpl::serialize
//
//================================================================

void FormatTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize(kit, STR("Display"), {STR("<Nothing>"), STR("")}, {STR("Format Test")});
    iterationCount.serialize(kit, STR("Iteration Count"));
}

//================================================================
//
// FormatTestImpl::process
//
//================================================================

stdbool FormatTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    ////

    printMsgL(kit, STR("Format test."));

    ////

    using Float = float32;

    constexpr int maxWidth = 0;
    constexpr int maxDigits = 6;
    constexpr int minExpo = -12;
    constexpr int maxExpo = +12;
    constexpr int maxPrecision = 12;

    for_count (iteration, iterationCount())
    {
        int width = randRange(rndgen, 0, maxWidth);
        int digits = randRange(rndgen, 0, maxDigits);

        uint64 bodyInt = 0;

        for_count (i, digits)
            bodyInt = bodyInt * 10 + randRange(rndgen, 0, 9);

        ////

        auto value = float64(bodyInt);
        int expo = randRange(rndgen, minExpo, maxExpo);
        value *= pow(10.0, expo - digits);

        ////

        int precision = randRange(rndgen, 0, maxPrecision);

        ////

        kit.formatter.clear();
        kit.formatter.write(fltf(Float(value), ' ', width, precision));
        REQUIRE(kit.formatter.valid());

        auto resultOpt = SimpleString{kit.formatter.str()};
        REQUIRE(def(resultOpt));

        ////

        if (iteration == 0)
            printMsgL(kit, STR("%"), resultOpt);

        ////

        SimpleString resultRef;

        auto writer = [&] (auto* ptr, auto size)
            {resultRef.append(ptr, size);};

        require(referenceFixedFormat(bodyInt, expo - digits, precision, writer, stdPass));

        REQUIRE(def(resultRef));

        ////

        if_not (resultRef == resultOpt)
        {
            resultRef.clear();
            require(referenceFixedFormat(bodyInt, expo - digits, precision, writer, stdPass));

            printMsgG(kit, STR("Ref %"), resultRef, msgWarn);
            printMsgG(kit, STR("Opt %"), resultOpt, msgWarn);
            printMsgG(kit, STR(""));
        }

    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
