#include "rndgen/randRange.h"
#include "numbers/safeint32/safeint32.h"

namespace safeint32TestImpl {

//================================================================
//
// TYPE_BIT_COUNT
//
//================================================================

#define TYPE_BIT_COUNT(Type) \
    (sizeof(Type) * CHAR_BIT)

//================================================================
//
// 
//
//================================================================

template <typename Perk>
bool typeTest1
(
    const typename Perk::Type sample[],
    int32 N,
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    ////

    for (int32 i = 0; i < N; ++i)
    {
        REQUIRE(bool(Perk::equal(Perk::opStd(sample[i]), Perk::opTest(sample[i]))));
    }

    ////

    stdEnd;
}

//================================================================
//
// 
//
//================================================================

template <typename Perk>
bool typeTest2
(
    const typename Perk::Type sampleA[],
    int32 countA,
    const typename Perk::Type sampleB[],
    int32 countB,
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    ////

    for (int32 iA = 0; iA < countA; ++iA)
    {
        for (int32 iB = 0; iB < countB; ++iB)
        {
            REQUIRE
            (
                bool // @ccfix
                (
                    Perk::equal
                    (
                        Perk::opStd(sampleA[iA], sampleB[iB]), 
                        Perk::opTest(sampleA[iA], sampleB[iB])
                    )
                )
            );
        }
    }

    ////

    stdEnd;
}

//================================================================
//
// Tests don't rely on assumption that int32 is stored in 2s complement code.
//
// The bit representation of a tested number is unsigned 32-bit,
// using 2s complementary code, 0x80000000 servers as NAN.
//
//================================================================

COMPILE_ASSERT(TYPE_BIT_COUNT(uint32) == TYPE_BIT_COUNT(int32));
COMPILE_ASSERT(TYPE_BIT_COUNT(uint32) <= DBL_MANT_DIG); // Can be represented in full precision

//================================================================
//
// 
//
//================================================================

template <typename Perk>
struct StdTest
{

    COMPILE_ASSERT(TYPE_EQUAL(TYPE_MAKE_UNCONTROLLED(Perk), int32));

    ////

    typedef uint32
        Type;

    ////

    static bool equal(uint32 A, uint32 B)
    {
        return A == B;
    }

};

//================================================================
//
// 
//
//================================================================

static const uint32 specialCases[] =
{
    // Near zero
    0xFFFFFFFCUL,
    0xFFFFFFFDUL,
    0xFFFFFFFEUL,
    0xFFFFFFFFUL,
    0x00000000UL,
    0x00000001UL,
    0x00000002UL,
    0x00000003UL,

    // Near 0x80000000
    0x7FFFFFFCUL,
    0x7FFFFFFDUL,
    0x7FFFFFFEUL,
    0x7FFFFFFFUL,
    0x80000000UL,
    0x80000001UL,
    0x80000002UL,
    0x80000003UL,

    // Near 0x0010000
    0x0000FFFCUL,
    0x0000FFFDUL,
    0x0000FFFEUL,
    0x0000FFFFUL,
    0x00010000UL,
    0x00010001UL,
    0x00010002UL,
    0x00010003UL,

    // Near 0x0008000
    0x00007FFCUL,
    0x00007FFDUL,
    0x00007FFEUL,
    0x00007FFFUL,
    0x00008000UL,
    0x00008001UL,
    0x00008002UL,
    0x00008003UL
};

//================================================================
//
// 
//
//================================================================

static const uint32 shiftCases[] =
{
    0xFFFFFFFDUL,
    0xFFFFFFFEUL,
    0xFFFFFFFFUL,
    0x00000000UL,
    0x00000001UL,
    0x00000002UL,
    0x00000003UL,
    0x00000004UL,
    0x00000005UL,
    0x00000006UL,
    0x00000007UL,
    0x00000008UL,
    0x00000009UL,
    0x0000000AUL,
    0x0000000BUL,
    0x0000000CUL,
    0x0000000DUL,
    0x0000000EUL,
    0x0000000FUL,
    0x00000010UL,
    0x00000011UL,
    0x00000012UL,
    0x00000013UL,
    0x00000014UL,
    0x00000015UL,
    0x00000016UL,
    0x00000017UL,
    0x00000018UL,
    0x00000019UL,
    0x0000001AUL,
    0x0000001BUL,
    0x0000001CUL,
    0x0000001DUL,
    0x0000001EUL,
    0x0000001FUL,
    0x00000020UL,
    0x00000021UL,
    0x00000022UL,
    0x00000023UL,
    0x7FFFFFFDUL,
    0x7FFFFFFEUL,
    0x7FFFFFFFUL,
    0x80000000UL,
    0x80000001UL,
    0x80000002UL,
    0x80000003UL
};

//================================================================
//
// 
//
//================================================================

sysinline uint32 nan() {return 0x80000000UL;}

sysinline double toDbl(uint32 X)
{
    double result = 0;

    if (X & 0x80000000UL)
        result = -double(~X + 1);
    else
        result = double(X);

    return result;
}

//----------------------------------------------------------------

sysinline uint32 fromDbl(double X)
{
    uint32 result = 0x80000000UL;

    if (X >= double(-0x7FFFFFFFL) && X <= double(+0x7FFFFFFFL))
    {
        if (X < 0)
            result = ~uint32(-X) + 1;
        else
            result = uint32(X);
    }

    return result;
}

//================================================================
//
// Unary plus and minus
//
//================================================================

template <typename Perk>
struct UnaryMinus : StdTest<Perk>
{

    static uint32 opStd(uint32 X)
    {
        uint32 result = nan();

        if (X != nan())
            result = fromDbl(-toDbl(X));

        return result;
    }

    static uint32 opTest(uint32 X)
    {
        return uint32(convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>(-convertExact<Perk>(int32(X))));
    }

};

//----------------------------------------------------------------

template <typename Perk>
struct UnaryPlus : StdTest<Perk>
{
    static uint32 opStd(uint32 X) 
        {return X;}
    

    static uint32 opTest(uint32 X) 
        {return uint32(convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>(+convertExact<Perk>(int32(X))));}
};

//----------------------------------------------------------------

template <typename Perk, typename Test>
bool testUnaryPlusMinus
(
    RndgenState& rndgen,
    stdPars(ErrorLogKit)
)
{
    stdBegin;

    uint32 allCases[COMPILE_ARRAY_SIZE(specialCases) + 128];

    for (int32 i = 0; i < COMPILE_ARRAY_SIZE(specialCases); ++i)
        allCases[i] = specialCases[i];

    for (int32 i = COMPILE_ARRAY_SIZE(specialCases); i < COMPILE_ARRAY_SIZE(allCases); ++i)
        allCases[i] = rand(rndgen, uint32(0x00000000UL), uint32(0xFFFFFFFFUL));

    require(typeTest1<Test>(allCases, COMPILE_ARRAY_SIZE(allCases), stdPass));

    stdEnd;
}

//================================================================
//
// Simple implementation of a binary operation:
// everything goes via double without any checks.
//
//================================================================

#define DEFINE_SIMPLE_BINARY_TRAIT(Name, OP) \
\
template <typename Perk> \
struct Name : StdTest<Perk> \
{ \
    \
    static uint32 opStd(uint32 A, uint32 B) \
    { \
        uint32 result = nan(); \
        \
        if (A != nan() && B != nan()) \
            result = fromDbl(toDbl(A) OP toDbl(B)); \
        \
        return result; \
    } \
    \
    static uint32 opTest(uint32 A, uint32 B) \
    { \
        return uint32 \
        ( \
            convertExact<TYPE_MAKE_UNCONTROLLED(Perk)> \
            ( \
                operator OP \
                ( \
                    convertExact<Perk>(int32(A)), \
                    convertExact<Perk>(int32(B)) \
                ) \
            ) \
        ); \
    } \
    \
};

//================================================================
//
// 
//
//================================================================

template <typename Perk, typename Test>
bool testBinarySpecialCasesAndRandom(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    ////

    uint32 allCases[COMPILE_ARRAY_SIZE(specialCases) + 64];

    for (int32 i = 0; i < COMPILE_ARRAY_SIZE(specialCases); ++i)
        allCases[i] = specialCases[i];

    for (int32 i = COMPILE_ARRAY_SIZE(specialCases); i < COMPILE_ARRAY_SIZE(allCases); ++i)
        allCases[i] = rand(rndgen, uint32(0x00000000UL), uint32(0xFFFFFFFFUL));

    ////

    require
    (
        typeTest2<Test>
        (
            allCases, COMPILE_ARRAY_SIZE(allCases),
            allCases, COMPILE_ARRAY_SIZE(allCases),
            stdPass
        )
    );

    stdEnd;
}

//================================================================
//
// Multiplication
//
//================================================================

COMPILE_ASSERT(DBL_MAX_EXP >= 2*32); // Can hold the result of 32-bit numbers product

DEFINE_SIMPLE_BINARY_TRAIT(Multiplication, *)

//----------------------------------------------------------------

template <typename Perk>
bool testMultiplication(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    ////

    typedef Multiplication<Perk> Test;

    require((testBinarySpecialCasesAndRandom<Perk, Test>(rndgen, stdPass)));

    //
    // Special test near overflow edge
    //

    for (int32 i = 0; i < 8192; ++i)
    {
        uint32 X = rand(rndgen, uint32(0x00000000UL), uint32(0x7FFFFFFFUL));

        double A = toDbl(X >> rand(rndgen, 0, 31));

        if (rand(rndgen, 0, 1) == 0)
            A = -A;

        if (A != 0)
        {
            double M = (rand(rndgen, 0, 1) == 0) ? double(-0x7FFFFFFFL) : double(+0x7FFFFFFFL);

            double Mdiv = M / A;

            double B0 = Mdiv >= 0 ? floor(Mdiv) : ceil(Mdiv);
            double B1 = B0 + (B0 >= 0 ? +1 : -1);

            uint32 A32u = fromDbl(A);

            uint32 B032u = fromDbl(B0);
            uint32 B132u = fromDbl(B1);

            REQUIRE(bool(Test::equal(Test::opStd(A32u, B032u), Test::opTest(A32u, B032u))));
            REQUIRE(bool(Test::equal(Test::opStd(A32u, B132u), Test::opTest(A32u, B132u))));

            double res0 = A * B0;
            bool ovf0 = !(res0 >= double(-0x7FFFFFFFL) && res0 <= double(+0x7FFFFFFFL));

            double res1 = A * B1;
            bool ovf1 = !(res1 >= double(-0x7FFFFFFFL) && res1 <= double(+0x7FFFFFFFL));

            REQUIRE(ovf0 != ovf1);
        }
    }

    ////

    stdEnd;
}

//================================================================
//
// Division
//
//================================================================

template <typename Perk>
struct Division : StdTest<Perk>
{

    static uint32 opStd(uint32 A, uint32 B)
    {
        uint32 result = nan();

        if (A != nan() && B != nan())
        {
            if (toDbl(B) != 0)
            {
                double r = toDbl(A) / toDbl(B);
                r = (r >= 0) ? floor(r) : ceil(r);
                result = fromDbl(r);
            }
        }

        return result;
    }

    static uint32 opTest(uint32 A, uint32 B)
    {
        return uint32
        (
            convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>
            (
                operator / // @ccfix
                (
                    convertExact<Perk>(int32(A)), 
                    convertExact<Perk>(int32(B))
                )
            )
        );
    }

};

//----------------------------------------------------------------

template <typename Perk>
bool testDivision(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    bool ok = testBinarySpecialCasesAndRandom<Perk, Division<Perk> >(rndgen, stdPass);
    require(ok);

    stdEnd;
}

//================================================================
//
// Remainder
//
//================================================================

template <typename Perk>
struct Remainder : StdTest<Perk>
{

    static uint32 opStd(uint32 A, uint32 B)
    {
        uint32 result = nan();

        if (A != nan() && B != nan())
        {
            if (toDbl(B) != 0)
            {
                double r = toDbl(A) / toDbl(B);
                r = (r >= 0) ? floor(r) : ceil(r);
                double rem = toDbl(A) - r * toDbl(B);
                result = fromDbl(rem);
            }
        }

        return result;
    }

    static uint32 opTest(uint32 A, uint32 B)
    {
        return uint32
        (
            convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>
            (
                operator % // @ccfix
                (
                    convertExact<Perk>(int32(A)),
                    convertExact<Perk>(int32(B))
                )
            )
        );
    }

};

//----------------------------------------------------------------

template <typename Perk>
bool testRemainder(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    bool ok = testBinarySpecialCasesAndRandom<Perk, Remainder<Perk> >
        (rndgen, stdPass);

    stdEndEx(ok);
}

//================================================================
//
// Addition
//
//================================================================

DEFINE_SIMPLE_BINARY_TRAIT(Addition, +)

//----------------------------------------------------------------

template <typename Perk>
bool testAddition(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    bool ok = testBinarySpecialCasesAndRandom<Perk, Addition<Perk> >
        (rndgen, stdPass);

    stdEndEx(ok);
}

//================================================================
//
// Subtraction
//
//================================================================

DEFINE_SIMPLE_BINARY_TRAIT(Subtraction, -)

//----------------------------------------------------------------

template <typename Perk>
bool testSubtraction(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    bool ok = testBinarySpecialCasesAndRandom<Perk, Subtraction<Perk> >
        (rndgen, stdPass);

    stdEndEx(ok);
}

//================================================================
//
// Shifts
//
//================================================================

template <typename Perk>
struct ShiftRight : StdTest<Perk>
{

    static uint32 opStd(uint32 A, uint32 B)
    {
        uint32 result = nan();

        if (A != nan() && B != nan())
        {
            double Bd = toDbl(B);
            

            if (Bd >= 0 && Bd <= 31)
            {
                double tmp = ldexp(toDbl(A), -int(Bd)); // @ccfix
                result = fromDbl(floor(tmp));
            }
        }

        return result;
    }

    static uint32 opTest(uint32 A, uint32 B)
    {
        return uint32
        (
            convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>
            (
                operator >> // @ccfix
                (
                    convertExact<Perk>(int32(A)),
                    convertExact<Perk>(int32(B))
                )
            )
        );
    }

};

//----------------------------------------------------------------

template <typename Perk>
struct ShiftLeft : StdTest<Perk>
{

    static uint32 opStd(uint32 A, uint32 B)
    {
        uint32 result = nan();

        if (A != nan() && B != nan())
        {
            double Bd = toDbl(B);
            

            if (Bd >= 0 && Bd <= 31)
            {
                double tmp = ldexp(toDbl(A), +int(Bd)); // @ccfix
                result = fromDbl(floor(tmp)); 
            }
        }

        return result;
    }

    static uint32 opTest(uint32 A, uint32 B)
    {
        return uint32
        (
            convertExact<TYPE_MAKE_UNCONTROLLED(Perk)>
            (
                operator << // @ccfix
                (
                    convertExact<Perk>(int32(A)),
                    convertExact<Perk>(int32(B))
                )
            )
        );
    }

};

//----------------------------------------------------------------

template <typename Perk, typename Test>
bool testShift(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    ////

    uint32 allCases[COMPILE_ARRAY_SIZE(specialCases) + 64];

    for (int32 i = 0; i < COMPILE_ARRAY_SIZE(specialCases); ++i)
        allCases[i] = specialCases[i];

    for (int32 i = COMPILE_ARRAY_SIZE(specialCases); i < COMPILE_ARRAY_SIZE(allCases); ++i)
        allCases[i] = rand(rndgen, uint32(0x00000000UL), uint32(0xFFFFFFFFUL));

    ////

    require
    (
        typeTest2<Test>
        (
            allCases, COMPILE_ARRAY_SIZE(allCases),
            shiftCases, COMPILE_ARRAY_SIZE(shiftCases),
            stdPass
        )
    );

    stdEnd;
}

//================================================================
//
// Burn-in operation test
//
//================================================================

template <typename Perk>
bool burninTest(RndgenState& rndgen, stdPars(ErrorLogKit))
{
    stdBegin;

    require((testUnaryPlusMinus<Perk, UnaryMinus<Perk> >(rndgen, stdPass)));
    require((testUnaryPlusMinus<Perk, UnaryPlus<Perk> >(rndgen, stdPass)));

    require(testMultiplication<Perk>(rndgen, stdPass));

    require(testDivision<Perk>(rndgen, stdPass));
    require(testRemainder<Perk>(rndgen, stdPass));

    require(testAddition<Perk>(rndgen, stdPass));
    require(testSubtraction<Perk>(rndgen, stdPass));

    require((testShift<Perk, ShiftLeft<Perk> >(rndgen, stdPass)));
    require((testShift<Perk, ShiftRight<Perk> >(rndgen, stdPass)));

    stdEnd;
}

//================================================================
//
// test
//
//================================================================

bool safeint32Test(stdPars(ErrorLogKit))
{
    stdBegin;

    RndgenState rndgen(0x8FA9E36D);
    require(burninTest<safeint32::Type>(rndgen, stdPass));

    stdEnd;
}

//----------------------------------------------------------------

}
