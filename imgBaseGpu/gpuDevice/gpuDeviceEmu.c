#include "gpuDeviceEmu.h"

#include "compileTools/compileLoop.h"
#include "data/matrix.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// emuThrowError
//
//================================================================

[[noreturn]]
void emuThrowError(EmuError error)
{
    throw error;
}

//================================================================
//
// EMU_REQUIRE
//
//================================================================

#define EMU_REQUIRE(condition) \
    if (condition) ; else emuThrowError(TRACE_AUTO_LOCATION CT(": ") CT(TRACE_STRINGIZE(condition)) CT(" failed"))

//================================================================
//
// warpPredicateImpl
//
//================================================================

uint32 warpPredicateImpl(bool pred, devPars)
{
    EmuParams& args = emuParams;

    ////

    auto threadIdx = args.fiberIdx;
    auto threadCount = areaOf(args.sharedParams->threadCount);

    ////

    auto& memory = args.sharedParams->warpIntrinsicsMemory;
    ARRAY_EXPOSE(memory);

    EMU_REQUIRE(SpaceU(threadCount) <= SpaceU(memorySize));

    ////

    devSyncThreads(); // Finish previous reading.

    ////

    EMU_REQUIRE(SpaceU(threadIdx) < SpaceU(threadCount));
    memoryPtr[threadIdx] = pred;

    ////

    devSyncThreads(); // Finish writing.

    ////

    auto warpIdx = Space(SpaceU(threadIdx) / SpaceU(devWarpSize));
    auto warpBase = warpIdx * devWarpSize;

    EMU_REQUIRE(warpBase + devWarpSize <= threadCount);

    ////

    uint32 value = 0;

    auto ptr = memoryPtr + warpBase;

    #define TMP_MACRO(i, _) \
        value += (uint32(ptr[i]) << i);

    COMPILE_LOOP(devWarpSize, devWarpSize, TMP_MACRO, )

    #undef TMP_MACRO

    ////

    return value;
}

//================================================================
//
// readLaneGeneric
//
//================================================================

template <typename Type, typename GetReadLane>
sysinline Type readLaneGeneric(const Type& value, const GetReadLane& getReadLane, devPars)
{
    EmuParams& args = emuParams;

    auto threadIdx = args.fiberIdx;
    auto threadCount = areaOf(args.sharedParams->threadCount);

    ////

    auto& memory = args.sharedParams->warpIntrinsicsMemory;
    ARRAY_EXPOSE(memory);

    EMU_REQUIRE(SpaceU(threadCount) <= SpaceU(memorySize));

    ////

    devSyncThreads(); // Finish previous reading.

    ////

    EMU_REQUIRE(SpaceU(threadIdx) < SpaceU(threadCount));
    recastFittingLayout<Type>(memoryPtr[threadIdx]) = value;

    ////

    devSyncThreads(); // Finish writing.

    ////

    auto warpIdx = Space(SpaceU(threadIdx) / SpaceU(devWarpSize));
    auto warpBase = warpIdx * devWarpSize;

    auto currentLane = threadIdx - warpBase;
    EMU_REQUIRE(SpaceU(currentLane) < SpaceU(devWarpSize));

    ////

    auto readLane = getReadLane(currentLane);
    EMU_REQUIRE(SpaceU(readLane) < SpaceU(devWarpSize));

    auto readIdx = warpBase + readLane;

    EMU_REQUIRE(SpaceU(readIdx) <= SpaceU(threadCount));
    return recastFittingLayout<Type>(memoryPtr[readIdx]);
}

//================================================================
//
// readLaneBorderWrapImpl
//
//================================================================

template <typename Type>
Type readLaneBorderWrapImpl(const Type& value, int lane, devPars)
{
    auto readLane = [&] (int32)
    {
        auto result = SpaceU(lane) % SpaceU(devWarpSize);
        return Space(result);
    };

    return readLaneGeneric(value, readLane, devPass);
}

//----------------------------------------------------------------

template <typename Type>
Type readLaneAddBorderClampImpl(const Type& value, int delta, devPars)
{
    auto readLane = [&] (int32 currentLane)
    {
        auto result = clampRange<int32>(currentLane + delta, 0, devWarpSize - 1);
        return result;
    };

    return readLaneGeneric(value, readLane, devPass);
}

//----------------------------------------------------------------

template <typename Type>
Type readLaneSubBorderClampImpl(const Type& value, int delta, devPars)
{
    auto readLane = [&] (int32 currentLane)
    {
        auto result = clampRange<int32>(currentLane - delta, 0, devWarpSize - 1);
        return result;
    };

    return readLaneGeneric(value, readLane, devPass);
}

//----------------------------------------------------------------

template <typename Type>
Type readLaneXorImpl(const Type& value, int32 mask, devPars)
{
    auto readLane = [&] (int32 currentLane)
    {
        return (uint32(currentLane) ^ uint32(mask)) % uint32(devWarpSize);
    };

    return readLaneGeneric(value, readLane, devPass);
}

//================================================================
//
// Instantiations.
//
//================================================================

#define TMP_MACRO(Type) \
    INSTANTIATE_FUNC_EX(readLaneBorderWrapImpl<Type>, readLaneBorderWrapImpl##Type) \
    INSTANTIATE_FUNC_EX(readLaneAddBorderClampImpl<Type>, readLaneAddBorderClampImpl##Type) \
    INSTANTIATE_FUNC_EX(readLaneSubBorderClampImpl<Type>, readLaneSubBorderClampImpl##Type) \
    INSTANTIATE_FUNC_EX(readLaneXorImpl<Type>, readLaneXorImpl##Type)

TMP_MACRO(int32)
TMP_MACRO(uint32)
TMP_MACRO(float32)

TMP_MACRO(int64)
TMP_MACRO(uint64)
TMP_MACRO(float64)

#undef TMP_MACRO
