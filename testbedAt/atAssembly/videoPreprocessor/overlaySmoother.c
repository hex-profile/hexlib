#include "overlaySmoother.h"

#include <process.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "storage/rememberCleanup.h"
#include "storage/classThunks.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTool.h"
#include "copyMatrixAsArray.h"
#include "flipMatrix.h"
#include "history/historyObj.h"
#include "dataAlloc/arrayMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "cfgTools/multiSwitch.h"
#include "userOutput/printMsgEx.h"
#include "errorLog/debugBreak.h"
#include "interfaces/threadManager.h"
#include "tfiltParam.h"
#include "timer/timer.h"
#include "timerImpl/timerImpl.h"
#include "data/spacex.h"

namespace overlaySmoother {

//================================================================
//
// queueCapacity
//
//================================================================

static const Space queueCapacity = 16;
static const Space averagingFrameCount = 32;

static const Space outputMonitorCount = 256;

//================================================================
//
// maxPlaybackDelay
//
//================================================================

static const float32 minPlaybackDelay = 1e-3f; // 1000 fps
static const float32 maxPlaybackDelay = 500e-3f; // 0.5 sec

//================================================================
//
// ColorPixel
//
//================================================================

using ColorPixel = uint8_x4;

//================================================================
//
// AtProviderFromCpuImage
//
//================================================================

class AtProviderFromCpuImage : public AtImageProvider<uint8_x4>
{

public:

    AtProviderFromCpuImage(const Matrix<ColorPixel>& cpuImage, const ErrorLogKit& kit)
        : cpuImage(cpuImage), kit(kit) {}

    bool saveImage(const Matrix<ColorPixel>& dest, stdNullPars);

    Space baseByteAlignment() const {return cpuBaseByteAlignment;}

    Space getPitch() const {return cpuImage.memPitch();}

private:

    Matrix<ColorPixel> cpuImage;
    ErrorLogKit kit;

};

//================================================================
//
// AtProviderFromCpuImage::saveImage
//
//================================================================

bool AtProviderFromCpuImage::saveImage(const Matrix<ColorPixel>& dest, stdNullPars)
{
    stdBegin;

    ////

    Matrix<const ColorPixel> src = cpuImage;
    Matrix<ColorPixel> dst = dest;

    ////

    REQUIRE(equalSize(src, dst));
    REQUIRE(src.memPitch() == dst.memPitch());

    ////

    if (src.memPitch() < 0)
    {
        src = flipMatrix(src);
        dst = flipMatrix(dst);
    }

    REQUIRE(src.memPitch() >= src.sizeX());

    ////

    Array<const ColorPixel> srcArray;
    require(getMatrixMemoryRangeAsArray(src, srcArray, stdPass));

    Array<ColorPixel> dstArray;
    require(getMatrixMemoryRangeAsArray(dst, dstArray, stdPass));

    ////

    ARRAY_EXPOSE_UNSAFE(srcArray, src);
    ARRAY_EXPOSE_UNSAFE(dstArray, dst);

    REQUIRE(srcSize == dstSize);
    memcpy(dstPtr, srcPtr, dstSize * sizeof(ColorPixel));

    ////

    stdEnd;
}

//================================================================
//
// QueueImage
//
//================================================================

struct QueueImage
{
    ArrayMemory<ColorPixel> memory;
    Matrix<ColorPixel> matrix;
};

//----------------------------------------------------------------

inline void exchange(QueueImage& a, QueueImage& b)
{
    exchange(a.memory, b.memory);
    exchange(a.matrix, b.matrix);
}

//================================================================
//
// saveImageToQueue
//
//================================================================

template <typename Kit>
bool saveImageToQueue(const Point<Space>& size, AtImageProvider<uint8_x4>& provider, QueueImage& dst, stdPars(Kit))
{
    stdBegin;

    Space desiredPitch = provider.getPitch();
    Space memPitch = absv(desiredPitch);
    REQUIRE(size.X <= memPitch);

    ////

    Space memSize = 0;
    REQUIRE(safeMul(memPitch, size.Y, memSize));

    ////

    if_not (memSize <= dst.memory.size())
        require(dst.memory.realloc(memSize, provider.baseByteAlignment(), kit.malloc, stdPass));

    ////

    REQUIRE(dst.memory.resize(memSize));

    ////

    ARRAY_EXPOSE_UNSAFE(dst.memory, dstMemory);

    Matrix<ColorPixel> dstMatrix(dstMemoryPtr, memPitch, size.X, size.Y);

    if (desiredPitch < 0)
        dstMatrix = flipMatrix(dstMatrix);

    ////

    require(provider.saveImage(dstMatrix, stdPass));
    dst.matrix = dstMatrix;

    ////

    stdEnd;
}

//================================================================
//
// SharedStruct
//
//================================================================

struct SharedStruct
{

    bool initialized = false;

    EventOwner serverWake; // no change at run time
    EventOwner clientWake; // no change at run time

    //
    // Shared variables
    //

    CriticalSection varLock;

    bool varShutdown = false;

    bool varRunning = true;

    bool running()
    {
        CRITSEC_GUARD(varLock);
        return varRunning;
    }

    bool varSmoothing = true;

    bool smoothing()
    {
        CRITSEC_GUARD(varLock);
        return varSmoothing;
    }

    float32 varTargetDelay;
    HistoryObjStatic<float32, outputMonitorCount> varActualDelays;
    HistoryObjStatic<float32, outputMonitorCount> varRenderDelays;

    //
    // Queue
    //

    CriticalSection queueLock; // queue lock first, output lock second!
    HistoryObjStatic<QueueImage, queueCapacity> queue;

    //
    // Output interface
    //

    CriticalSection outputLock;
    AtAsyncOverlay* outputInterface;

    ////

    SharedStruct()
    {
        varTargetDelay = 0;
        DEBUG_BREAK_CHECK(varActualDelays.reallocStatic(outputMonitorCount));
        DEBUG_BREAK_CHECK(varRenderDelays.reallocStatic(outputMonitorCount));
        outputInterface = 0;
    }

    //----------------------------------------------------------------
    //
    // Deinit
    //
    //----------------------------------------------------------------

    void deinit()
    {

        varLock.clear();
        queueLock.clear();
        outputLock.clear();

        ////

        serverWake.clear();
        clientWake.clear();

        ////

        varShutdown = false;
        varRunning = false;

        ////

        varSmoothing = true;
        varTargetDelay = 0;
        varActualDelays.clear();
        varRenderDelays.clear();
        outputInterface = 0;

        initialized = false;
    }

    //----------------------------------------------------------------
    //
    // init
    //
    //----------------------------------------------------------------

    bool init(stdPars(InitKit))
    {
        stdBegin;

        if (initialized)
            return true;

        ////

        SharedStruct& self = *this;
        REMEMBER_CLEANUP1_EX(totalCleanup, self.deinit(), SharedStruct&, self);

        ////

        require(kit.threadManager.createCriticalSection(varLock, stdPass));
        require(kit.threadManager.createCriticalSection(queueLock, stdPass));
        require(kit.threadManager.createCriticalSection(outputLock, stdPass));

        ////

        require(kit.threadManager.createEvent(false, serverWake, stdPass));
        require(kit.threadManager.createEvent(false, clientWake, stdPass));

        ////

        require(queue.realloc(queueCapacity, stdPass));
        queue.clear();

        ////

        varShutdown = false;
        varRunning = true;

        ////

        varSmoothing = true;
        varTargetDelay = 0;
        varActualDelays.clear();
        varRenderDelays.clear();
        outputInterface = 0;

        ////

        totalCleanup.cancel();
        initialized = true;

        stdEnd;
    }

};

//================================================================
//
// ErrorLogDebugBreak
//
//================================================================

class ErrorLogDebugBreak : public ErrorLog
{

public:

    bool isThreadProtected() const override
        {return true;}

    void addErrorSimple(const CharType* message) override
        {DEBUG_BREAK_INLINE();}

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) override
        {DEBUG_BREAK_INLINE();}

};

//================================================================
//
// tryToOutputOneFrame
//
// Return bool: hard error (stop thread);
//
//================================================================

bool tryToOutputOneFrame(SharedStruct& shared, bool& lastOutputDefined, TimeMoment& lastOutput, Timer& timer, float32& resultWaitTime, uint32 outputFrameCount, stdPars(ErrorLogKit))
{
    stdBegin;

    //----------------------------------------------------------------
    //
    // Check if need to wait for output
    //
    //----------------------------------------------------------------

    if (lastOutputDefined)
    {
        TimeMoment currentMoment = timer.moment();
        float32 elapsedTime = timer.diff(lastOutput, currentMoment);

        ////

        bool targetSmoothing = false;
        float32 targetDelay = 0;

        {
            CRITSEC_GUARD(shared.varLock);
            targetSmoothing = shared.varSmoothing;
            targetDelay = shared.varTargetDelay;
        }

        ////

        float32 timeToWait = clampMin(targetDelay - elapsedTime, 0.f);

        if (targetSmoothing && timeToWait > 0)
            {resultWaitTime = timeToWait; return true;}
    }

    //----------------------------------------------------------------
    //
    // Try to output one frame
    //
    //----------------------------------------------------------------

    {
        CRITSEC_GUARD(shared.queueLock);
        CRITSEC_GUARD(shared.outputLock);

        Space queueSize = shared.queue.size();

        bool everythingIsReady =
            queueSize >= 1 && shared.outputInterface != 0;

        if_not (everythingIsReady)
            {resultWaitTime = float32Nan(); return true;}

        ////

        QueueImage* image = shared.queue[queueSize-1];
        REQUIRE(image != 0);

        ////

        AtProviderFromCpuImage provider(image->matrix, kit);

        TimeMoment renderBegin = timer.moment();
        REQUIRE(shared.outputInterface->setImage(image->matrix.size(), provider, stdPass));
        TimeMoment renderEnd = timer.moment();

        ////

        {
            CRITSEC_GUARD(shared.varLock);

            if (lastOutputDefined && outputFrameCount >= 8)
            {
                float32 lastDelay = timer.diff(lastOutput, renderBegin);

                if (lastDelay <= maxPlaybackDelay)
                    *shared.varActualDelays.add() = lastDelay;

                *shared.varRenderDelays.add() = timer.diff(renderBegin, renderEnd);
            }
        }

        lastOutputDefined = true;
        lastOutput = renderBegin;

        ////

        shared.queue.removeOldest();
        shared.clientWake.set();
    }

    ////

    resultWaitTime = 0;
    stdEnd;
}

//================================================================
//
// serverFunc
//
//================================================================

void serverFunc(void* param)
{

    SharedStruct& shared = * (SharedStruct*) param;

    //
    // Abort handler: set flag and wake (potentially) waiting client.
    //

    REMEMBER_CLEANUP1_EX
    (
        signalAbort,
        {
            {
                CRITSEC_GUARD(shared.varLock);
                shared.varRunning = false;
            }

            shared.clientWake.set();
        },
        SharedStruct&, shared
    );

    ////

    TRACE_ROOT_STD;

    ////

    ErrorLogDebugBreak errorLog;
    ErrorLogKit kit(errorLog, 0);

    ////

    TimerImpl timer;

    TimeMoment lastOutput;
    bool lastOutputDefined = false;

    uint32 outputFrameCount = 0;

    ////

    for (;;)
    {

        {
            CRITSEC_GUARD(shared.varLock);

            if (shared.varShutdown)
                break;
        }

        ////

        float32 resultWaitTime = 0;
        requirev(tryToOutputOneFrame(shared, lastOutputDefined, lastOutput, timer, resultWaitTime, outputFrameCount, stdPass));

        if (resultWaitTime == 0.f)
        {
            ++outputFrameCount;
            continue; // go next frame
        }

        ////

        if_not (def(resultWaitTime))
        {
            shared.serverWake.wait();
        }
        else
        {
            REQUIREV(resultWaitTime >= 0.f);
            int32 waitMs = convertDown<int32>(resultWaitTime * 1000);

            if (waitMs > 0)
                shared.serverWake.waitWithTimeout(waitMs);
        }

    }

    ////

    signalAbort.cancel(); // normal shutdown
}

//================================================================
//
// OverlaySmootherImpl
//
//================================================================

class OverlaySmootherImpl
{

public:

    OverlaySmootherImpl()
        :
        inputDelayFilter(0.f),
        queueOccupancyFilter(0.f),
        cfgMaxOutputRate(1.f/maxPlaybackDelay, 1.f/minPlaybackDelay, 60.f)
    {
    }

    ~OverlaySmootherImpl() {deinit();}

public:

    void serialize(const ModuleSerializeKit& kit);

public:

    bool init(stdPars(InitKit));
    void deinit();

public:

    void setOutputInterface(AtAsyncOverlay* output);

public:

    virtual bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdPars(ProcessKit));
    virtual bool updateImage(stdPars(ProcessKit));
    virtual bool clearQueue(stdPars(ProcessKit));
    virtual bool setSmoothing(bool smoothing, stdPars(ProcessKit));
    virtual bool flushSmoothly(stdPars(ProcessKit));

private:

    bool initialized = false;
    ThreadControl workerThread;

    ////

    QueueImage temporaryBuffer;

    ////

    SharedStruct shared;

    ////

    BoolVarStatic<true> displayFrameRate;
    BoolVarStatic<false> displayInfo;
    NumericVar<float32> cfgMaxOutputRate;

    ////

    bool lastInputDefined = false;
    TimeMoment lastInputMoment;

    ////

    TfiltParam<float32> inputDelayFilter;
    TfiltParam<float32> queueOccupancyFilter;

};

//================================================================
//
// computeQueueBoostFactor
//
//================================================================

inline float32 computeQueueBoostFactor(float32 maxRecentQueueOccupancy)
{
    return 1.f + 0.25f * (1 - maxRecentQueueOccupancy);
}

//================================================================
//
// computeTargetDelay
//
//================================================================

inline float32 computeTargetDelay(float32 avgInputDelay, float32 maxRecentQueueOccupancy, float32 maxRate)
{
    float32 boostedDelay = avgInputDelay * computeQueueBoostFactor(maxRecentQueueOccupancy);

    float32 renderInstabilityAdjustment = 2e-3f;
    float32 renderAdjustedDelay = avgInputDelay + renderInstabilityAdjustment;

    float32 minDelay = minPlaybackDelay;

    if (maxRate > 0)
        minDelay = clampRange(1.f / maxRate, minPlaybackDelay, maxPlaybackDelay);

    return clampRange(maxv(boostedDelay, renderAdjustedDelay), minDelay, maxPlaybackDelay);
}

//================================================================
//
// OverlaySmootherImpl::deinit
//
//================================================================

void OverlaySmootherImpl::deinit()
{

    if_not (initialized)
        return;

    //
    // Shutdown the thread
    //

    {
        CRITSEC_GUARD(shared.varLock);

        shared.varShutdown = true;
    }

    shared.serverWake.set();

    workerThread.waitAndClear();

    //
    // Deinit
    //

    shared.deinit();

    lastInputDefined = false;
    inputDelayFilter.initialize(0.f);
    queueOccupancyFilter.initialize(0.f);

    initialized = false;

}

//================================================================
//
// OverlaySmootherImpl::init
//
//================================================================

bool OverlaySmootherImpl::init(stdPars(InitKit))
{
    stdBegin;

    deinit();

    //
    // Shared struct
    //

    require(shared.init(stdPass));
    REMEMBER_CLEANUP1_EX(sharedCleanup, shared.deinit(), SharedStruct&, shared);

    shared.varSmoothing = true;
    shared.varTargetDelay = computeTargetDelay(1.f / cfgMaxOutputRate, 0.f, cfgMaxOutputRate);

    //
    // Launch thread (should be the last)
    //

    require(kit.threadManager.createThread(serverFunc, &shared, 65536, workerThread, stdPass));
    REQUIRE(workerThread->setPriority(ThreadPriorityPlus1));

    ////

    sharedCleanup.cancel();

    lastInputDefined = false;
    inputDelayFilter.initialize(1.f / cfgMaxOutputRate);
    queueOccupancyFilter.initialize(0.f);

    initialized = true;

    stdEnd;
}

//================================================================
//
// OverlaySmootherImpl::serialize
//
//================================================================

void OverlaySmootherImpl::serialize(const ModuleSerializeKit& kit)
{
    cfgMaxOutputRate.serialize(kit, STR("Max Output Frame Rate"));

    displayFrameRate.serialize(kit, STR("Display Frame Rate"));
    displayInfo.serialize(kit, STR("Display Info"));
}

//================================================================
//
// OverlaySmootherImpl::setOutputInterface
//
//================================================================

void OverlaySmootherImpl::setOutputInterface(AtAsyncOverlay* output)
{
    requirev(initialized && shared.running());

    {
        CRITSEC_GUARD(shared.outputLock);
        shared.outputInterface = output;

        shared.serverWake.set();
    }
}

//================================================================
//
// OverlaySmootherImpl::setImage
//
//================================================================

bool OverlaySmootherImpl::setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdPars(ProcessKit))
{
    stdBegin;

    require(initialized && shared.running());

    ////

    if (textEnabled)
        printMsg(kit.localLog, STR("OVERLAY: %0"), desc);

    //----------------------------------------------------------------
    //
    // Copy the image into the temporary buffer
    //
    //----------------------------------------------------------------

    require(saveImageToQueue(size, imageProvider, temporaryBuffer, stdPass));

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    TimeMoment beginMoment = kit.timer.moment();

    //----------------------------------------------------------------
    //
    // Wait for the place in the queue
    //
    //----------------------------------------------------------------

    for (;;)
    {
        {
            CRITSEC_GUARD(shared.queueLock);

            if (shared.queue.size() < queueCapacity)
                break; // found place
        }

        shared.clientWake.wait();

        require(shared.running() != 0); // if worker aborted, return fail
    }

    TimeMoment stuckEndMoment = kit.timer.moment();

    //----------------------------------------------------------------
    //
    // Input delay filter
    //
    //----------------------------------------------------------------

    float32 lastInputDelay = 0;

    if (lastInputDefined)
    {
        lastInputDelay = kit.timer.diff(lastInputMoment, beginMoment);

        if (shared.smoothing() && lastInputDelay <= maxPlaybackDelay)
            REQUIRE(inputDelayFilter.add(lastInputDelay, float32(averagingFrameCount), float32(averagingFrameCount)));
    }

    ////

    lastInputDefined = true;
    lastInputMoment = stuckEndMoment;

    //----------------------------------------------------------------
    //
    // Queue size filter
    //
    //----------------------------------------------------------------

    Space queueSize = 0;

    {
        CRITSEC_GUARD(shared.queueLock);
        queueSize = shared.queue.size();
    }

    ////

    float32 currentQueueOccupancy = clampRange(float32(queueSize) / (queueCapacity - 1), 0.f, 1.f);
    REQUIRE(queueOccupancyFilter.add(currentQueueOccupancy, 1.f, float32(averagingFrameCount)));

    //----------------------------------------------------------------
    //
    // Update target delay
    //
    //----------------------------------------------------------------

    float32 targetDelay = computeTargetDelay(inputDelayFilter(), queueOccupancyFilter(), cfgMaxOutputRate);

    {
        CRITSEC_GUARD(shared.varLock);
        shared.varTargetDelay = targetDelay;
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    float32 minActualDelay = typeMax<float32>();
    float32 maxActualDelay = 0;

    {
        CRITSEC_GUARD(shared.varLock);

        Space count = shared.varActualDelays.size();

        for (Space i = 0; i < count; ++i)
        {
            float32 value = *shared.varActualDelays[i];
            minActualDelay = minv(minActualDelay, value);
            maxActualDelay = maxv(maxActualDelay, value);
        }

    }

    minActualDelay = clampMax(minActualDelay, maxActualDelay);

    float32 actualJitter = maxActualDelay - minActualDelay;

    //----------------------------------------------------------------
    //
    // Render jitter
    //
    //----------------------------------------------------------------

    float32 minRenderDelay = typeMax<float32>();
    float32 maxRenderDelay = 0;

    {
        CRITSEC_GUARD(shared.varLock);

        Space count = shared.varRenderDelays.size();

        for (Space i = 0; i < count; ++i)
        {
            float32 value = *shared.varRenderDelays[i];
            minRenderDelay = minv(minRenderDelay, value);
            maxRenderDelay = maxv(maxRenderDelay, value);
        }

    }

    minRenderDelay = clampMax(minRenderDelay, maxRenderDelay);

    float32 renderJitter = maxRenderDelay - minRenderDelay;

    //----------------------------------------------------------------
    //
    // Diagnostics
    //
    //----------------------------------------------------------------

    if (textEnabled && displayFrameRate && shared.smoothing())
    {
        printMsgL(kit, STR("OverlaySmoother: Input %0 fps, Output %1 fps, Jitter %2 ms (%3%%)"),
            fltf(1.f / inputDelayFilter(), 1),
            fltf(1.f / targetDelay, 1),
            fltf(actualJitter * 1000, 1),
            fltf(actualJitter / minActualDelay * 100, 0));
    }

    if (textEnabled && displayInfo)
    {

        printMsgL(kit, STR("OverlaySmoother: Input Momentary %0 ms, Input Smooth %1 ms, Target %2 ms"),
            fltf(lastInputDelay * 1e3f, 1),
            fltf(inputDelayFilter() * 1e3f, 1),
            fltf(targetDelay * 1e3f, 1));

        printMsgL(kit, STR("OverlaySmoother: Render Delay [%0, %1] ms, Jitter %2 ms"),
            fltf(minRenderDelay * 1e3f, 1), fltf(maxRenderDelay * 1e3f, 1),
            fltf((maxRenderDelay - minRenderDelay) * 1e3f, 1));

        printMsgL(kit, STR("OverlaySmoother: Output Delay [%0, %1] ms, Jitter %2%%"),
            fltf(minActualDelay * 1e3f, 1), fltf(maxActualDelay * 1e3f, 1),
            fltf((maxActualDelay / minActualDelay - 1.f) * 100, 0));

        printMsgL(kit, STR("OverlaySmoother: Queue Size = %0 of %1 (Filtered %2, Boost %3)"), queueSize, queueCapacity,
            fltf(queueOccupancyFilter() * queueCapacity, 1), fltf(computeQueueBoostFactor(queueOccupancyFilter()), 2));

    }

    //----------------------------------------------------------------
    //
    // Put the image into the queue
    //
    //----------------------------------------------------------------

    {
        CRITSEC_GUARD(shared.queueLock);

        REQUIRE(shared.queue.size() < queueCapacity);

        QueueImage* image = shared.queue.add();
        REQUIRE(image != 0);
        exchange(temporaryBuffer, *image);
    }

    //----------------------------------------------------------------
    //
    // Wake worker thread
    //
    //----------------------------------------------------------------

    shared.serverWake.set();

    ////

    stdEnd;
}

//================================================================
//
// OverlaySmootherImpl::updateImage
//
//================================================================

bool OverlaySmootherImpl::updateImage(stdPars(ProcessKit))
{
    stdBegin;
    stdEnd;
}

//================================================================
//
// OverlaySmootherImpl::clearQueue
//
//================================================================

bool OverlaySmootherImpl::clearQueue(stdPars(ProcessKit))
{
    stdBegin;

    require(initialized && shared.running());

    {
        CRITSEC_GUARD(shared.queueLock);
        shared.queue.clear();
    }

    stdEnd;
}

//================================================================
//
// OverlaySmootherImpl::setSmoothing
//
//================================================================

bool OverlaySmootherImpl::setSmoothing(bool smoothing, stdPars(ProcessKit))
{
    stdBegin;

    require(initialized && shared.running());

    {
        CRITSEC_GUARD(shared.varLock);
        shared.varSmoothing = smoothing;
    }

    if_not (smoothing)
    {
        CRITSEC_GUARD(shared.queueLock);

        if_not (smoothing)
            shared.queue.clear();
    }

    stdEnd;
}

//================================================================
//
// OverlaySmootherImpl::flushSmoothly
//
//================================================================

bool OverlaySmootherImpl::flushSmoothly(stdPars(ProcessKit))
{
    stdBegin;

    require(initialized && shared.running());

    //----------------------------------------------------------------
    //
    // Wait for the empty queue
    //
    //----------------------------------------------------------------

    for (;;)
    {
        {
            CRITSEC_GUARD(shared.queueLock);

            if (shared.queue.size() == 0)
                break;
        }

        shared.clientWake.wait();

        require(shared.running() != 0); // if worker aborted, return fail
    }

    ////

    stdEnd;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(OverlaySmoother)

CLASSTHUNK_VOID1(OverlaySmoother, serialize, const ModuleSerializeKit&)

CLASSTHUNK_BOOL_STD0(OverlaySmoother, init, InitKit);
CLASSTHUNK_VOID0(OverlaySmoother, deinit);

CLASSTHUNK_BOOL_STD5(OverlaySmoother, setImage, const Point<Space>&, AtImageProvider<uint8_x4>&, const FormatOutputAtom&, uint32, bool, ProcessKit);
CLASSTHUNK_BOOL_STD0(OverlaySmoother, updateImage, ProcessKit);
CLASSTHUNK_BOOL_STD0(OverlaySmoother, clearQueue, ProcessKit);
CLASSTHUNK_BOOL_STD1(OverlaySmoother, setSmoothing, bool, ProcessKit);
CLASSTHUNK_BOOL_STD0(OverlaySmoother, flushSmoothly, ProcessKit);

CLASSTHUNK_VOID1(OverlaySmoother, setOutputInterface, AtAsyncOverlay*)

//----------------------------------------------------------------

}
