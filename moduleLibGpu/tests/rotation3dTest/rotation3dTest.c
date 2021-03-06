#include "rotation3dTest.h"

#include "cfgTools/multiSwitch.h"
#include "storage/classThunks.h"
#include "userOutput/printMsgEx.h"
#include "rndgen/rndgenFloat.h"
#include "timer/timer.h"
#include "mathFuncs/rotationMath3d.h"
#include "mathFuncs/rotationMath.h"
#include "errorLog/errorLog.h"
#include "storage/rememberCleanup.h"
#include "rndgen/rndgenPoint.h"

//================================================================
//
// quatMixedMinus
//
// The quaternions should have unit length.
//
// Returns D such that it moves a point from A to B.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatMixedMinus(const Point4D<Float>& B, const Point4D<Float>& A)
{
    return quatToRodrigues(B % ~A);
}

//================================================================
//
// quatMixedPlus
//
// The quaternion should have unit length.
//
// If D is movement from A to B, the operation returns B.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatMixedPlus(const Point4D<Float>& A, const Point3D<Float>& D)
{
    return quatFromRodrigues(D) % A; // Apply A, then apply D.
}

//================================================================
//
// quatL2Diff
//
//================================================================

template <typename Float>
sysinline Float quatL2Diff(const Point4D<Float>& A, const Point4D<Float>& B)
{
    Float lenSq1 = vectorLengthSq(A - B);
    Float lenSq2 = vectorLengthSq(A + B);
    return fastSqrt(minv(lenSq1, lenSq2));
}

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

    RndgenState rndgen = 0;

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
        {STR("Rotation 3D Test"), STR("Ctrl+Shift+Alt+R")}
    );
}

//================================================================
//
// Rotation3DTestImpl::process
//
//================================================================

stdbool Rotation3DTestImpl::process(stdPars(GpuModuleProcessKit))
{
    DisplayType displayType = kit.verbosity >= Verbosity::On ? displaySwitch : DisplayNothing;

    if (displayType == DisplayNothing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Test
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP_EX(failMessage, printMsgG(kit, STR("Rotation 3D Math Test FAILED"), msgErr));

    ////

    if (rndgen == 0)
        rndgen = RndgenState(kit.timer.convertToSteadyMicroseconds(kit.timer.moment()));

    ////

    float32 maxDelta = pi32 * 0.999f;
    float32 eps = 0.001f * maxDelta;

    ////

    auto generateUnitQuat = [&] () 
    {
        return vectorNormalize(rndgenGaussApproxFour<Point4D<float32>>(rndgen));
    };

    auto generateUnitVec = [&] ()
    {
        return vectorNormalize(rndgenGaussApproxFour<Point3D<float32>>(rndgen));
    };

    auto generateMapVec = [&] ()
    {
        return rndgenUniformFloat(rndgen, maxDelta) * generateUnitVec();
    };

    auto generateSomeVec = [&] ()
    {
        return rndgenUniformFloat(rndgen, 5.0f) * generateUnitVec();
    };

    ////

    Space testCount = 1024;

    for_count (i, testCount)
    {

        //----------------------------------------------------------------
        //
        // Test (Q + D) - Q == D
        //
        //----------------------------------------------------------------

        {
            auto Q = generateUnitQuat();
            auto D = generateMapVec();

            auto DErr = quatMixedMinus(quatMixedPlus(Q, D), Q) - D;
            require(vectorLength(DErr) <= eps);
        }

        //----------------------------------------------------------------
        //
        // Test Q + (P - Q) == P
        //
        //----------------------------------------------------------------

        {
            auto Q = generateUnitQuat();
            auto P = generateUnitQuat();

            auto Ptest = quatMixedPlus(Q, quatMixedMinus(P, Q));
            auto err = quatL2Diff(P, Ptest);
            require(err <= eps);
        }

        //----------------------------------------------------------------
        //
        // Test ||(Q + D1) - (Q + D2)|| <= ||D1 - D2||
        //
        //----------------------------------------------------------------

        {
            auto Q = generateUnitQuat();
            auto D1 = generateMapVec();
            auto D2 = generateMapVec();

            auto len1 = vectorLength(quatMixedMinus(quatMixedPlus(Q, D1), quatMixedPlus(Q, D2)));
            auto len2 = vectorLength(D1 - D2);
            require(len1 <= len2 + eps);
        }

        //----------------------------------------------------------------
        //
        // Test rotation... somehow.
        //
        //----------------------------------------------------------------

        {
            auto vec = generateSomeVec();
            auto A = generateUnitQuat();
            auto B = generateUnitQuat();

            // Rotate forward and backward.
            require(vectorLength(~A % (A % vec) - vec) <= eps);

            // Combine two rotations.
            auto v1 = B % (A % vec);
            auto v2 = (B % A) % vec;
            require(vectorLength(v1 - v2) <= eps);
        }

        //----------------------------------------------------------------
        //
        // Test rotation by matrix.
        //
        //----------------------------------------------------------------

        {
            auto Q = generateUnitQuat();
            auto V = generateSomeVec();

            auto M = quatMat(Q);

            auto v1 = Q % V;
            auto v2 = M % V;

            require(vectorLength(v1 - v2) <= eps);

            ////

            v1 = ~Q % V;
            v2 = ~M % V;

            require(vectorLength(v1 - v2) <= eps);
        }

        //----------------------------------------------------------------
        //
        // Test equivalence of double-cover.
        //
        //----------------------------------------------------------------

        {
            auto Q = generateUnitQuat();
            auto P = generateUnitQuat();
            auto D = generateMapVec();
            auto vec = generateSomeVec();

            require(quatL2Diff(+Q, -Q) <= eps);

            // [+]
            require(quatL2Diff(quatMixedPlus(+Q, D), quatMixedPlus(-Q, D)) <= eps);

            // [-]
            require(vectorLength(quatMixedMinus(+Q, P) - quatMixedMinus(-Q, P)) <= eps);

            // Rotate vec
            require(vectorLength((+Q % vec) - (-Q % vec)) <= eps);

            // Quat mul
            require(quatL2Diff(+Q % P, -Q % P) <= eps);
            require(quatL2Diff(Q % -P, Q % +P) <= eps);
        }

    }

    ////

    failMessage.cancel();
    printMsgL(kit, STR("Rotation 3D Math Test OK"));

    ////

    returnTrue;
}
