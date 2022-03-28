#if HOSTCODE
#include "gpuOverheadTest.h"
#endif

#if HOSTCODE
#include <stdio.h>
#if HEXLIB_PLATFORM == 1
#include <cuda.h>
#endif
#endif

#include "gpuDevice/gpuDevice.h"
#include "prepTools/prepFor.h"

#if HOSTCODE
#include "cfg/cfgInterface.h"
#include "storage/rememberCleanup.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "dataAlloc/arrayMemory.h"
#include "timer/timer.h"
#include "userOutput/printMsgEx.h"
#include "numbers/mathIntrinsics.h"
#include "rndgen/rndgenFloat.h"
#include "numbers/divRound.h"
#include "errorLog/debugBreak.h"
#endif

//================================================================
//
// Model
//
//================================================================

/*

For Kepler:

0.0099 + 1.98e-6 * groupCount, for <= 11 warps
0.0099 + 4.08e-6 * groupCount, for <= 22 warps
0.0098 + 9.71e-6 * groupCount, otherwise

*/

//================================================================
//
// EmptyKernelParams
//
//================================================================

struct EmptyKernelParams {};

//================================================================
//
// KERNEL_COUNT
//
//================================================================

#define KERNEL_COUNT 50

//================================================================
//
// emptyKernel
//
//================================================================

#if DEVCODE

#define TMP_MACRO(i, _) \
    devDefineKernel(PREP_PASTE(emptyKernel, i), EmptyKernelParams, params) {}

PREP_FOR(KERNEL_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

#endif

//----------------------------------------------------------------

#if HOSTCODE

    static const GpuKernelLink* emptyKernelArray[KERNEL_COUNT] = 
    {
        #define TMP_MACRO(i, _) &emptyKernel##i,
        PREP_FOR(KERNEL_COUNT, TMP_MACRO, _)
        #undef TMP_MACRO
    };

#endif

//================================================================
//
// GpuOverheadTest::serialize
//
//================================================================

#if HOSTCODE

void GpuOverheadTest::serialize(const ModuleSerializeKit& kit)
{
    active.serialize(kit, STR("Is Active"));
    fixedGroupSize.serialize(kit, STR("Fixed Group Size"));
    fixedImageRatio.serialize(kit, STR("Fixed Image Ratio"));
    fixedGroupWarps.serialize(kit, STR("Fixed Group Warps"));
    reliabilityFactor.serialize(kit, STR("Reliability Factor"));
}

#endif

//================================================================
//
// GpuOverheadTest::process
//
//================================================================

#if HOSTCODE

stdbool GpuOverheadTest::process(stdPars(ProcessKit))
{
    if_not (active) 
        returnTrue;

#if 0 && HEXLIB_PLATFORM == 1 // After stream implementation was changed, the test is broken

    ////

    RndgenState r = rndgenState;

    Space testCount = clampMin(5000 / reliabilityFactor, 1);

    ////

    require(gpuSyncCurrentStream(stdPass));

    const GpuStream& stream = kit.gpuCurrentStream; 
    require(kit.gpuStreamWaiting.waitStream(stream, stdPass));

    ////

    CUevent cuStartEvent = 0;
    REQUIRE(cuEventCreate(&cuStartEvent, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
    REMEMBER_CLEANUP(DEBUG_BREAK_CHECK(cuEventDestroy(cuStartEvent) == CUDA_SUCCESS));

    CUevent cuStopEvent = 0;
    REQUIRE(cuEventCreate(&cuStopEvent, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
    REMEMBER_CLEANUP(DEBUG_BREAK_CHECK(cuEventDestroy(cuStopEvent) == CUDA_SUCCESS));

    //----------------------------------------------------------------
    //
    // Measure kernel calls
    //
    //----------------------------------------------------------------

    #define ALLOCATE_TEST_ARRAY(name, Type) \
        ARRAY_ALLOC(name, Type, testCount); \
        ARRAY_EXPOSE(name);

    ALLOCATE_TEST_ARRAY(groupSize, Point<Space>)
    ALLOCATE_TEST_ARRAY(groupCount, Point<Space>)

    ////

    Space maxGroupSize = kit.gpuProperties.maxGroupArea;
    Space warpSize = devWarpSize;
    REQUIRE(maxGroupSize % warpSize == 0);

    Space minGroupWarps = 2;
    Space maxGroupWarps = 8;

    if (fixedGroupSize)
        minGroupWarps = maxGroupWarps = fixedGroupWarps;

    Space warpSizeBits = convertNearest<Space>(nativeLog2(float32(warpSize)));

    float32 minWarpCountBits = nativeLog2(float32(minGroupWarps));
    float32 maxWarpCountBits = nativeLog2(float32(maxGroupWarps));
 
    ////

    float32 minImageArea = float32(32*32);
    float32 maxImageArea = float32(3840*2160);

    ////

    for_count (i, testCount)
    {
        Space warpCountBits = convertNearest<Space>(linerp<float32>(rndgenUniformFloat(r), minWarpCountBits, maxWarpCountBits));
        Space groupSizeBits = warpCountBits + warpSizeBits;

        Space groupSizeBitsX = convertNearest<Space>(linerp<float32>(rndgenUniformFloat(r), 0.f, float32(groupSizeBits)));
        Space groupSizeBitsY = groupSizeBits - groupSizeBitsX;

        ////

        Space groupSizeX = 1 << groupSizeBitsX;
        Space groupSizeY = 1 << groupSizeBitsY;
        Point<Space> groupSize = point(groupSizeX, groupSizeY);

        ////

        float32 imageArea = linerp(rndgenUniformFloat(r), minImageArea, maxImageArea);
        float32 imageAreaBits = nativeLog2(imageArea);

        float32 ratioAreaBits = clampMax(4.f, imageAreaBits);

        if (fixedImageRatio)
            ratioAreaBits = 0;

        float32 ratioBitsX = linerp<float32>(rndgenUniformFloat(r), 0.f, ratioAreaBits);
        float32 ratioBitsY = ratioAreaBits - ratioBitsX;

        float32 fixedAreaBits = imageAreaBits - ratioAreaBits;
        float32 imageBitsX = 0.5f * fixedAreaBits + ratioBitsX;
        float32 imageBitsY = 0.5f * fixedAreaBits + ratioBitsY;

        Space imageSizeX = clampMin(convertNearest<Space>(nativePow2(imageBitsX)), 1);
        Space imageSizeY = clampMin(convertNearest<Space>(nativePow2(imageBitsY)), 1);
        Point<Space> imageSize = point(imageSizeX, imageSizeY);

        ////

        Point<Space> groupCount = clampRange(divUpNonneg(imageSize, groupSize), point(1), point(0xFFFF));
        Point<Space> actualImageSize = groupCount * groupSize;

        ////

        if (kit.dataProcessing)
        {
            groupSizePtr[i] = groupSize;
            groupCountPtr[i] = groupCount;
        }
    }

    //----------------------------------------------------------------
    //
    // Measurements arrays
    //
    //----------------------------------------------------------------

    ALLOCATE_TEST_ARRAY(deviceTimeCombined, float32);
    ALLOCATE_TEST_ARRAY(hostTime, float32);

    if (kit.dataProcessing)
    {
        for_count (i, testCount)
        {
            deviceTimeCombinedPtr[i] = 0;
            hostTimePtr[i] = 0;
        }
    }

    ////

    for_count (i, testCount)
    {
        //
        // Combined mega-call
        //

        float32 deviceTimeCombined = 0;
        float32 hostTime = 0;

        if (kit.dataProcessing)
        {
            TimeMoment t1 = kit.timer.moment();
            REQUIRE(cuEventRecord(cuStartEvent, stream) == CUDA_SUCCESS);

            for_count (k, reliabilityFactor)
            {
                const GpuKernelLink* emptyKernel = emptyKernelArray[rndgen16(r) % KERNEL_COUNT];
                require(kit.gpuKernelCalling.callKernel(groupCountPtr[i], groupSizePtr[i], 0, *emptyKernel, EmptyKernelParams(), kit.gpuCurrentStream, stdPass));
            }

            REQUIRE(cuEventRecord(cuStopEvent, stream) == CUDA_SUCCESS);
            REQUIRE(cuEventSynchronize(cuStopEvent) == CUDA_SUCCESS);
            TimeMoment t2 = kit.timer.moment();

            float32 cudaTimeMs = 0;
            REQUIRE(cuEventElapsedTime(&cudaTimeMs, cuStartEvent, cuStopEvent) == CUDA_SUCCESS);
            float32 deviceTime = 1e-3f * cudaTimeMs;

            hostTimePtr[i] = kit.timer.diff(t1, t2) / reliabilityFactor;
            deviceTimeCombinedPtr[i] = deviceTime / reliabilityFactor;
        }
    }

    //----------------------------------------------------------------
    //
    // Accumulated measurements
    //
    //----------------------------------------------------------------

    if (0)
    {

        ALLOCATE_TEST_ARRAY(deviceTimeAccumulated, float32);

        if (kit.dataProcessing)
        {
            for_count (i, testCount)
                deviceTimeAccumulatedPtr[i] = 0;
        }

        ////

        for_count (t, reliabilityFactor)
        {
            ALLOCATE_TEST_ARRAY(order, Space);

            if (kit.dataProcessing)
            {
                for_count (i, testCount)
                    orderPtr[i] = i;

                for_count (i, testCount)
                {
                    Space j = rndgen16(r) % testCount;
                    exchange(orderPtr[i], orderPtr[j]);
                }

                for_count (q, testCount)
                {
                    Space i = orderPtr[q];

                    const GpuKernelLink* emptyKernel = emptyKernelArray[rndgen16(r) % KERNEL_COUNT];

                    REQUIRE(cuEventRecord(cuStartEvent, stream) == CUDA_SUCCESS);

                    require(kit.gpuKernelCalling.callKernel(groupCountPtr[i], groupSizePtr[i], 0, *emptyKernel, EmptyKernelParams(), kit.gpuCurrentStream, stdPass));

                    REQUIRE(cuEventRecord(cuStopEvent, stream) == CUDA_SUCCESS);
                    REQUIRE(cuEventSynchronize(cuStopEvent) == CUDA_SUCCESS);

                    float32 cudaTimeMs = 0;
                    REQUIRE(cuEventElapsedTime(&cudaTimeMs, cuStartEvent, cuStopEvent) == CUDA_SUCCESS);
                    float32 deviceTime = 1e-3f * cudaTimeMs;

                    deviceTimeAccumulatedPtr[i] += (1.f/reliabilityFactor) * deviceTime;
                }

            }
        }

    }

    //----------------------------------------------------------------
    //
    // measurementTime
    //
    //----------------------------------------------------------------
  
    ALLOCATE_TEST_ARRAY(measurementTime, float32);

    if (kit.dataProcessing)
    {
        for_count (i, testCount)
            measurementTimePtr[i] = deviceTimeCombinedPtr[i];
    }

    //----------------------------------------------------------------
    //
    // Save to file
    //
    //----------------------------------------------------------------

    uint32 startWriteIndex = 4;

    if (kit.dataProcessing && runCount() >= startWriteIndex)
    {
        const char* fileName = "D:\\gpuTime.txt";

        FILE* f = fopen(fileName, runCount() == startWriteIndex ? "wt" : "at");
        REQUIRE(f != 0);
        REMEMBER_CLEANUP(fclose(f));

        for_count (i, testCount)
        {
            Point<Space> groupSize = groupSizePtr[i];
            Point<Space> groupCount = groupCountPtr[i];

            fprintf(f, "%.0f\t %.0f\t %.0f\t %.0f\t %.0f\t %.0f\t   %.0f\t  %.10f\t\n", 
                float32(groupSize.X), float32(groupSize.Y), 
                float32(areaOf(groupSize)),
                float32(groupCount.X), float32(groupCount.Y), 
                float32(areaOf(groupCount)),
                float32(areaOf(groupSize)) * float32(areaOf(groupCount)),
                measurementTimePtr[i] * 1e3f
            );
        }

        writeCount += testCount;
        printMsgL(kit, STR("gpuOverheadTest: Writing to %0, %1 records"), fileName, writeCount(), msgWarn);
    }

    ////

    if (kit.dataProcessing) rndgenState = r;
    if (kit.dataProcessing) runCount++;

    ////

#endif

    returnTrue;
}

#endif
