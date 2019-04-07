#include "videoPreprocessor.h"

#include "flipMatrix.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "mathFuncs/rotationMath.h"
#include "dataAlloc/arrayMemory.inl"
#include "gpuImageVisualization/gpuImageConsoleImpl.h"
#include "cfgTools/numericVar.h"
#include "cfgTools/rangeValueControl.h"
#include "storage/classThunks.h"
#include "cfgTools/boolSwitch.h"
#include "history/historyObj.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "copyMatrixAsArray.h"
#include "storage/rememberCleanup.h"
#include "atAssembly/halfFloatTest/halfFloatTest.h"
#include "atAssembly/videoPreprocessor/tools/rotateImage.h"
#include "atAssembly/videoPreprocessor/tools/videoPrepTools.h"
#include "atAssembly/videoPreprocessor/displayWaitController.h"


#include "kits/userPoint.h"
#include "rndgen/rndgenFloat.h"
#include "timer/timer.h"
#include "cfgTools/multiSwitch.h"
#include "outImgAvi/outImgAvi.h"
#include "configFile/cfgSimpleString.h"
#include "randomImage/randomImage.h"
#include "kits/displayParamsKit.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsgEx.h"
#include "atAssembly/gpuImageConsoleAt/gpuImageConsoleAt.h"

#define USE_OVERLAY_SMOOTHER 1

#if USE_OVERLAY_SMOOTHER
    #include "atAssembly/videoPreprocessor/overlaySmoother.h"
#endif

////

namespace videoPreprocessor {

using namespace gpuImageConsoleImpl;

//================================================================
//
// maxFrameHistoryCapacity
//
//================================================================

static const Space maxFrameHistoryCapacity = 16;

//================================================================
//
// FrameSnapshot
//
//================================================================

struct FrameSnapshot
{
    GpuMatrixMemory<uint8_x4> frameMemory;
    GpuMatrix<uint8_x4> frame;
};

//================================================================
//
// ProcessCheck
// ProcessReset
//
//================================================================

struct ProcessCheck : public ProcessInspector
{
    bool allSteady = true;

    void operator()(bool& steadyProcessing)
        {if_not (steadyProcessing) allSteady = false;}
};

//----------------------------------------------------------------

struct ProcessReset : public ProcessInspector
{
    void operator()(bool& steadyProcessing)
        {steadyProcessing = true;}
};

//================================================================
//
// AtOverlayMonitor
//
//================================================================

class AtOverlayMonitor : public AtVideoOverlay
{

public:

    bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
        {overlayIsSet = true; return base.setImage(size, imageProvider, desc, id, textEnabled, stdNullPassThru);}

    bool setFakeImage(stdNullPars)
        {overlayIsSet = true; return base.setFakeImage(stdNullPassThru);}

    bool updateImage(stdNullPars)
        {return base.updateImage(stdNullPassThru);}

public:

    AtOverlayMonitor(AtVideoOverlay& base)
        : base(base) {}

public:

    bool overlayIsSet = false;

private:

    AtVideoOverlay& base;

};

//================================================================
//
// AviOutputConfig
//
//================================================================

struct AviOutputConfig
{

public:

    //
    // Output to AVI
    //

    BoolSwitch<false> savingActive;
    NumericVarStatic<int32, 1, 1024, 60> outputFps;
    CharArray outputDirName() {return STR("Output Directory");}
    SimpleStringVar outputDir;
    SimpleStringVar outputCodec;
    NumericVarStatic<int32, 0, 0x7FFFFFFF, 0> maxSegmentFrames;

public:

    AviOutputConfig()
    {
        outputDir = CT("");
        outputCodec = CT("DIB ");
    }

public:

    void serialize(const ModuleSerializeKit& kit)
    {
        savingActive.serialize(kit, STR("Active"));
        outputFps.serialize(kit, STR("Playback FPS"), STR("Playback framerate specified in AVI header"));

        kit.visitor(kit.scope, SerializeSimpleString(outputDir, outputDirName(), STR("Use double backslashes, for example C:\\\\Temp")));
        kit.visitor(kit.scope, SerializeSimpleString(outputCodec, STR("Compressor FourCC"), STR("Use 'DIB ' for uncompressed, 'ffds' for ffdshow")));

        maxSegmentFrames.serialize(kit, STR("Max Video Segment Frames"));
    }

};

//================================================================
//
// ProcessExKit
//
//================================================================

KIT_COMBINE4(ProcessExKit, ProcessKit, AlternativeVersionKit, DisplayParamsKit, GpuImageConsoleKit);

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// VideoPreprocessorImpl
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

class VideoPreprocessorImpl
{

public:

    VideoPreprocessorImpl();

    void serialize(const ModuleSerializeKit& kit);

    bool reallocValid() const;
    bool realloc(const Point<Space>& frameSize, stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;

    bool processEntry(VideoPrepTarget& target, stdPars(ProcessKit));

    bool processSingleFrame
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    bool processPrepFrontend
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    bool processCropFrontend
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    bool processTarget
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

private:

    Point<Space> allocFrameSize = point(0);

    ////

    NumericVarStatic<Space, 1, maxFrameHistoryCapacity, 4> frameHistoryCapacity;
    using FrameHistory = HistoryObjStatic<FrameSnapshot, maxFrameHistoryCapacity>;
    FrameHistory frameHistory;

private:

    BoolSwitch<false> displayFrameSize;

    ////

    bool prepParamsSteady = true;

    ////

    BoolSwitch<false> cropMode;
    NumericVar< Point<Space> > cropSizeCfg;

    ////

    enum GenMode {GenNone, GenPulse, GenGrating, GenRandom, GenModeCount};
    RingSwitch<GenMode, GenModeCount, GenNone> genMode;

    RangeValueControl<float32> genGratingPeriod;
    BoolSwitch<false> genGratingRectangleShape;

    NumericVarStatic<Space, 1, 1 << 20, 256> genPulsePeriod;

    ////

    RangeValueControl<float32> rotationAngle;

    uint32 movingFrameIndex = 0;

    ////

    StandardSignal randomizeSignal;

    BoolSwitch<false> simuMotionOn;
    NumericVarStaticEx<Point<float32>, int, -512, +512, 0> simuMotionSpeedCfg;

    Point<float32> simuMotionSpeed() const {return !simuMotionOn ? point(0.f) : simuMotionSpeedCfg;}

    sysinline bool simuMotionActive() const {return !allv(simuMotionSpeed() == 0.f);}

    //
    // Noise addition
    //

    BoolSwitch<false> noiseActive;
    NumericVarStaticEx<float32, int, 0, 1, 0> noiseSigma;

private:

    halfFloatTest::HalfFloatTest halfFloatTest;

private:

    RangeValueControl<float32> userDisplayFactor;
    MultiSwitch<VectorDisplayMode, VectorDisplayModeCount, VectorDisplayColor> vectorDisplayMode;
    DisplayWaitController displayWaitController;

    enum DisplayType {DisplayNothing, DisplayFrameHistory, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0xB0C09C28> displaySwitch;

private:

    BoolSwitch<false> alternativeVersion;

    RangeValueControl<Space> displayedTemporalIndex;
    RangeValueControl<int32> displayedCircularIndex;
    MultiSwitch<DisplaySide, DisplaySide_Count, DisplayOld> displayedSide;
    RangeValueControl<Space> displayedScaleIndex;
    RingSwitch<DisplayMethod, DISPLAY_METHOD_COUNT, DISPLAY_FULLSCREEN> displayMethod;

private:

#if USE_OVERLAY_SMOOTHER
    BoolSwitch<true> overlaySmootherEnabled;
    bool overlaySmootherTried = false;
    bool overlaySmootherInit = false;
    overlaySmoother::OverlaySmoother overlaySmoother;
#endif

private:

    AviOutputConfig aviConfig;
    OutImgAvi outImgAvi;

private:

    bool rndgenFrameInitialized = false;
    GpuMatrixMemory<RndgenState> rndgenFrame;

};

//================================================================
//
// VideoPreprocessorImpl::VideoPreprocessorImpl
//
//================================================================

VideoPreprocessorImpl::VideoPreprocessorImpl()
    :
    rotationAngle(0, 1, 0, 1.f/128, RangeValueCircular),
    cropSizeCfg(point(0), point(8192), point(1280, 720)),
    genGratingPeriod(2, 2048, 6, 1.02189714865411668f, RangeValueLogscale),
    userDisplayFactor(1.f/65536.f, 65536.f, 1.f, sqrtf(sqrtf(sqrtf(2))), RangeValueLogscale),

    displayedTemporalIndex(-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear),
    displayedCircularIndex(-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear),
    displayedScaleIndex(0, 0x7F, 0, 1, RangeValueLinear)

{
    noiseSigma = 0.01f;
}

//================================================================
//
// VideoPreprocessorImpl::serialize
//
//================================================================

void VideoPreprocessorImpl::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE_MODULE("Display Params");

        check_flag(alternativeVersion.serialize(kit, STR("Alternative Version"), STR("a")), prepParamsSteady);

        ////

        displayedTemporalIndex.serialize(kit, STR("Displayed Time Index"), STR(","), STR("."));
        displayedCircularIndex.serialize(kit, STR("Displayed Circular Index"), STR(";"), STR("'"));
        displayedScaleIndex.serialize(kit, STR("Displayed Scale Index"), STR("="), STR("-"));
        displayMethod.serialize(kit, STR("Display Method"), STR("Ctrl+D"));

        ////

        displayedSide.serialize
        (
            kit, STR("Displayed Direction"),
            {STR("Forward"), STR("9")},
            {STR("Backward"), STR("0")}
        );
    }

    {
        CFG_NAMESPACE_MODULE("Input Settings");

        displaySwitch.serialize
        (
            kit, STR("Display"),
            {STR("<Nothing>"), STR("")},
            {STR("Frame History"), STR("Alt+F")}
        );

        frameHistoryCapacity.serialize(kit, STR("Frame History Size"));
        displayFrameSize.serialize(kit, STR("Display Frame Size"), STR("Ctrl+F"));

        check_flag(cropMode.serialize(kit, STR("Crop Mode"), STR("Ctrl+C")), prepParamsSteady);
        check_flag(cropSizeCfg.serialize(kit, STR("Crop Size")), prepParamsSteady);
        check_flag(genMode.serialize(kit, STR("Generation Mode"), STR("Ctrl+G")), prepParamsSteady);
        check_flag(rotationAngle.serialize(kit, STR("Rotation Angle"), STR("PgUp"), STR("PgDn"), STR("End")), prepParamsSteady);
        check_flag(genGratingPeriod.serialize(kit, STR("Grating Period"), STR("Del"), STR("Ins")), prepParamsSteady);
        check_flag(genGratingRectangleShape.serialize(kit, STR("Grating Has Rectangle Shape")), prepParamsSteady);

        check_flag(genPulsePeriod.serialize(kit, STR("Pulse Period")), prepParamsSteady);

        randomizeSignal.serialize(kit, STR("Randomize"), STR("F1"));

        check_flag(simuMotionOn.serialize(kit, STR("Simulated Motion On"), STR("Ctrl+M")), prepParamsSteady);
        check_flag(simuMotionSpeedCfg.serialize(kit, STR("Simulated Motion Speed")), prepParamsSteady);

        check_flag(noiseActive.serialize(kit, STR("Noise Active"), STR("Ctrl+N")), prepParamsSteady);
        check_flag(noiseSigma.serialize(kit, STR("Noise Sigma")), prepParamsSteady);
    }

    {
        CFG_NAMESPACE_MODULE("Output Settings");

        {
            CFG_NAMESPACE_MODULE("Display Delayer");
            displayWaitController.serialize(kit);
        }

        userDisplayFactor.serialize(kit, STR("User Display Factor"), STR("Num +"), STR("Num -"), STR("Num *"));

        vectorDisplayMode.serialize
        (
            kit, STR("Vector Gray Mode"),
            {STR("Vector Display: Color"), STR("Z")},
            {STR("Vector Display: Magnitude"), STR("X")},
            {STR("Vector Display: X"), STR("C")},
            {STR("Vector Display: Y"), STR("V")}
        );

    }

    {
    #if USE_OVERLAY_SMOOTHER

        CFG_NAMESPACE_MODULE("Overlay Smoother");
        overlaySmootherEnabled.serialize(kit, STR("@Enabled"), STR("Alt+O"));
        overlaySmoother.serialize(kit);

    #endif
    }

    {
        CFG_NAMESPACE_MODULE("Saving AVI Files");
        aviConfig.serialize(kit);
    }
}

//================================================================
//
// VideoPreprocessorImpl::reallocValid
//
//================================================================

bool VideoPreprocessorImpl::reallocValid() const
{
    return
        frameHistoryCapacity == frameHistory.allocSize() &&
        halfFloatTest.reallocValid();
}

//================================================================
//
// VideoPreprocessorImpl::realloc
//
//================================================================

bool VideoPreprocessorImpl::realloc(const Point<Space>& frameSize, stdPars(ReallocKit))
{
    stdBegin;

    ////

    allocFrameSize = point(0);

    ////

    require(frameHistory.realloc(frameHistoryCapacity, stdPass));

    ////

    for (Space k = 0; k < frameHistoryCapacity; ++k)
    {
        FrameSnapshot* f = frameHistory.add();
        REQUIRE(f->frameMemory.realloc(frameSize, stdPass));
        f->frame = f->frameMemory;
    }

    frameHistory.clear();
    movingFrameIndex = 0;

    ////

    require(halfFloatTest.realloc(stdPass));

    ////

    require(rndgenFrame.realloc(frameSize, stdPass));
    rndgenFrameInitialized = false;

    ////

    allocFrameSize = frameSize;

    stdEnd;
}

//================================================================
//
// VideoPreprocessorImpl::outputFrameSize
//
//================================================================

Point<Space> VideoPreprocessorImpl::outputFrameSize() const
{
    return !cropMode ? allocFrameSize : clampMax(cropSizeCfg(), allocFrameSize);
}

//================================================================
//
// VideoPreprocessorImpl::processTarget
//
//================================================================

bool VideoPreprocessorImpl::processTarget
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdBeginScoped;

    //----------------------------------------------------------------
    //
    // Set AVI writer parameters
    //
    //----------------------------------------------------------------

    breakBlock(outAviOk)
    {
        if_not (aviConfig.savingActive)
            breakFalse;

        if_not (aviConfig.outputDir->length() != 0)
        {
            printMsgL(kit, STR("AVI Saving: <%0> is not set (Testbed->Config->Edit)"), aviConfig.outputDirName(), msgWarn);
            breakFalse;
        }

        breakBlock(setOk)
        {
            breakRequire(outImgAvi.setOutputDir(aviConfig.outputDir->cstr(), stdPass));
            breakRequire(outImgAvi.setCodec(outImgAvi::codecFromStr(aviConfig.outputCodec->cstr()), stdPass));
            breakRequire(outImgAvi.setFps(aviConfig.outputFps, stdPass));
            breakRequire(outImgAvi.setMaxSegmentFrames(aviConfig.maxSegmentFrames, stdPass));
        }

        if_not (setOk)
        {
            printMsgL(kit, STR("AVI Saving: Error happened"), msgWarn);
            breakFalse;
        }
    }

    if (outAviOk)
        printMsgL(kit, STR("AVI Saving: Files are saved to %0 (playback %1 fps, compressor '%2')"),
            aviConfig.outputDir->cstr(), aviConfig.outputFps(), aviConfig.outputCodec->cstr());

    ////

    AtImgConsole* atImageConsole = &kit.atImgConsole;
    AtVideoOverlay* atVideoOverlay = &kit.atVideoOverlay;

    ////

    OutImgAviThunk outAviThunk(outImgAvi, *atImageConsole, *atVideoOverlay, kit);

    if (outAviOk)
        {atImageConsole = &outAviThunk; atVideoOverlay = &outAviThunk;}

    //----------------------------------------------------------------
    //
    // GPU image console
    //
    //----------------------------------------------------------------

    GpuProhibitedConsoleThunk gpuBaseConsoleProhibited(kit);
    GpuBaseAtConsoleThunk gpuBaseConsoleAt(*atImageConsole, *atVideoOverlay, kit);
    GpuBaseConsole* gpuBaseConsole = &gpuBaseConsoleProhibited;

    if (kit.outputLevel >= OUTPUT_ENABLED)
        gpuBaseConsole = &gpuBaseConsoleAt;

    GpuImageConsoleThunk gpuImageConsole(*gpuBaseConsole, userDisplayFactor, vectorDisplayMode, kit);

    //----------------------------------------------------------------
    //
    // Display params
    //
    //----------------------------------------------------------------

    DisplayedRangeIndex displayedScaleIndexVar(displayedScaleIndex);
    DisplayedRangeIndex displayedTimeIndexVar(displayedTemporalIndex);

    DisplayParamsKit displayKit(inputFrame.size(), displayedSide, displayedTimeIndexVar, displayedScaleIndexVar,
        DisplayedCircularIndex(displayedCircularIndex), displayMethod);

    REMEMBER_CLEANUP2(displayedScaleIndex = displayedScaleIndexVar, RangeValueControl<Space>&, displayedScaleIndex, DisplayedRangeIndex&, displayedScaleIndexVar);
    REMEMBER_CLEANUP2(displayedTemporalIndex = displayedTimeIndexVar, RangeValueControl<Space>&, displayedTemporalIndex, DisplayedRangeIndex&, displayedTimeIndexVar);

    ProcessKit oldKit = kit;
    ProcessExKit kit = kitCombine(oldKit, GpuImageConsoleKit(gpuImageConsole, userDisplayFactor), displayKit, AlternativeVersionKit(alternativeVersion, 0));

    //----------------------------------------------------------------
    //
    // Display frame history
    //
    //----------------------------------------------------------------

    DisplayType displayType = kit.outputLevel >= OUTPUT_RENDER ? displaySwitch : DisplayNothing;

    if (displayType == DisplayFrameHistory)
    {
        if (frameHistory.size() == 0)
            printMsgL(kit, STR("Video Preprocessor: Frame history empty"), msgErr);
        else
        {
            Space i = kit.displayedTemporalIndex(-(frameHistory.size()-1), 0);

            require(kit.gpuImageConsole.addRgbColorImage(makeConst(frameHistory[-i]->frameMemory),
                0x00, 0xFF * kit.displayFactor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO,
                paramMsg(STR("Video Preprocessor: Frame history [%0]"), i), stdPass));
        }
    }

    //----------------------------------------------------------------
    //
    // Call target
    //
    //----------------------------------------------------------------

    DisplayDelayerThunk displayDelayer(displayWaitController, kit);

    require(target.process(stdPassKit(kitCombine(kit, GpuRgbFrameKit(inputFrame, 0), DisplayDelayerKit(displayDelayer, 0)))));

    ////

    stdEndScoped;
}

//================================================================
//
// VideoPreprocessorImpl::processSingleFrame
//
//================================================================

bool VideoPreprocessorImpl::processSingleFrame
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdBeginScoped;

    //----------------------------------------------------------------
    //
    // Overlay smoother
    //
    //----------------------------------------------------------------

#if USE_OVERLAY_SMOOTHER

    auto oldKit = kit;

    if_not (overlaySmootherTried)
    {
        overlaySmootherInit = overlaySmoother.init(stdPass);
        overlaySmoother.setOutputInterface(&kit.atAsyncOverlay);
        overlaySmootherTried = true;
    }

    require(overlaySmootherInit);

    ////

    overlaySmoother::OverlaySmootherThunk overlaySmootherThunk(overlaySmoother, kit);

    auto kit = kitReplace(oldKit, AtVideoOverlayKit(overlaySmootherThunk, 0));

    ////

    bool useOverlaySmoothing = overlaySmootherEnabled && (kit.atRunning && kit.atPlaying);

    require(overlaySmootherThunk.setSmoothing(useOverlaySmoothing, stdPass));

#endif

    //----------------------------------------------------------------
    //
    // Call
    //
    //----------------------------------------------------------------

    // ~~~ not exception-safe
    bool ok = processPrepFrontend(target, inputFrame, frameIndex, stdPass);

    //----------------------------------------------------------------
    //
    // Overlay
    //
    //----------------------------------------------------------------

#if USE_OVERLAY_SMOOTHER

    if_not (overlaySmootherThunk.overlayIsSet)
    {
        if (kit.dataProcessing && kit.outputLevel >= OUTPUT_ENABLED)
        {
            AtProviderFromGpuImage imageProvider(inputFrame, kit);
            require(overlaySmootherThunk.setImage(inputFrame.size(), imageProvider, STR("Input Frame"), 0, true, stdPass));
        }
    }

    ////

    if_not (useOverlaySmoothing)
        require(overlaySmootherThunk.flushSmoothly(stdPass));

#endif

    ////

    stdEndExScoped(ok);
}

//================================================================
//
// VideoPreprocessorImpl::processPrepFrontend
//
//================================================================

bool VideoPreprocessorImpl::processPrepFrontend
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdBegin;

    Point<Space> frameSize = inputFrame.size();

    AtOverlayMonitor atOverlayMonitor(kit.atVideoOverlay);
    ProcessKit kitMonitorEx = kitReplace(kit, AtVideoOverlayKit(atOverlayMonitor, 0));

    //----------------------------------------------------------------
    //
    // Direct pass-thru (no preprocessing)
    //
    //----------------------------------------------------------------

    bool prepOff =
        genMode == GenNone &&
        rotationAngle == 0 &&
        !simuMotionActive() &&
        !(noiseActive && noiseSigma > 0);

    if (prepOff)
    {
        require(processCropFrontend(target, inputFrame, frameIndex, stdPassKit(kitMonitorEx)));
        return true;
    }

    //----------------------------------------------------------------
    //
    // Rotate input frame
    //
    //----------------------------------------------------------------

    if (genMode == GenGrating)
        printMsgL(kit, STR("Grating Period = %0"), fltf(genGratingPeriod(), 2), msgWarn);

    if (rotationAngle())
        printMsgL(kit, STR("Rotation Angle = %0 (%1 deg)"), fltf(rotationAngle(), 3), fltf(rotationAngle*360, 1), msgWarn);

    if (simuMotionActive())
        printMsgL(kit, STR("Simulated %0 Speed = %1"), STR("Motion"), fltfs(simuMotionSpeed(), 2), msgWarn);

    ////

    Point<float32> forwardRotation = conjugate(circleCCW(rotationAngle()));

    ////

    GPU_MATRIX_ALLOC(processedFrameMemory, uint8_x4, frameSize);
    GpuMatrix<uint8_x4> processedFrame = processedFrameMemory;

    {
        GpuMatrix<const uint8_x4> srcFrame = inputFrame;
        GpuMatrix<uint8_x4> dstFrame = processedFrameMemory;
        Point<float32> usedRotation = forwardRotation;

        ////

        bool srcInvert = srcFrame.memPitch() < 0;

        if (srcInvert)
        {
            srcFrame = flipMatrix(srcFrame);
            processedFrame = flipMatrix(processedFrameMemory);
            usedRotation = conjugate(usedRotation);
        }

        ////

        Point<float32> transMul;
        Point<float32> transAdd;
        centerMatchedSpaceRotation(frameSize, frameSize, usedRotation, transMul, transAdd);

        Point<float32> motionOfs = float32(frameIndex) * simuMotionSpeed();

        if (srcInvert)
            motionOfs.Y = -motionOfs.Y;

        transAdd -= motionOfs;

        ////

        if (genMode == GenPulse)
        {
            require(generatePulse(processedFrameMemory, convertNearest<Space>(-motionOfs), genPulsePeriod, stdPass));
        }
        else if (genMode == GenGrating)
        {
            require(generateGrating(processedFrameMemory, genGratingPeriod, transMul, transAdd, genGratingRectangleShape(), stdPass));
        }
        else if (genMode == GenRandom)
        {
            require(initializeRandomStateMatrix(rndgenFrame, frameIndex, 0x113716C3, stdPass));
            require(generateRandom(processedFrameMemory, rndgenFrame, stdPass));
        }
        else
        {
            require
            (
                (simuMotionActive() ? rotateImageCubicMirror : rotateImageCubicZero)
                (srcFrame, processedFrameMemory, transMul, transAdd, stdPass)
            );
        }

        ////

        if (noiseActive && noiseSigma)
        {
            printMsgL(kit, STR("Adding input noise, sigma = %0"), fltf(noiseSigma(), 3), msgWarn);

            require(initializeRandomStateMatrix(rndgenFrame, frameIndex, 0x94D5B0B2, stdPass));
            require(generateAdditionalGaussNoise(processedFrameMemory, processedFrameMemory, rndgenFrame, noiseSigma, stdPass));
        }
    }

    //----------------------------------------------------------------
    //
    // Call main processing
    //
    //----------------------------------------------------------------

    require(processCropFrontend(target, processedFrame, frameIndex, stdPassKit(kitMonitorEx)));

    //----------------------------------------------------------------
    //
    // If no video overlay, set the rotated frame
    //
    //----------------------------------------------------------------

    if (kit.outputLevel >= OUTPUT_ENABLED && !atOverlayMonitor.overlayIsSet)
    {
        if (kit.dataProcessing)
        {
            AtProviderFromGpuImage imageProvider(processedFrame, kit);
            require(kit.atVideoOverlay.setImage(processedFrame.size(), imageProvider, STR("Rotated Frame"), 0, true, stdPass));
        }
    }

    ////

    stdEnd;
}

//================================================================
//
// VideoPreprocessorImpl::processCropFrontend
//
//================================================================

bool VideoPreprocessorImpl::processCropFrontend
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdBegin;

    ////

    Point<Space> frameSize = inputFrame.size();
    Point<Space> cropSize = clampRange(cropSizeCfg(), point(0), frameSize);

    ////

    if (!cropMode || allv(cropSize == frameSize))
    {
        require(processTarget(target, inputFrame, frameIndex, stdPass));
        return true;
    }

    ////

    // printMsgL(kit, STR("Cropping %0"), cropSize);

    ////

    Point<Space> cropOfs = (frameSize - cropSize) >> 1;

    //
    // Input frame
    //

    GPU_MATRIX_ALLOC(croppedFrameMemory, uint8_x4, cropSize);
    GpuMatrix<uint8_x4> croppedFrame = flipMatrix(croppedFrameMemory);
    require(copyImageRect(inputFrame, cropOfs, croppedFrame, stdPass));

    //
    // User point
    //

    bool userPointValid = kit.userPoint.valid;
    Point<Space> userPoint = kit.userPoint.position;

    if_not (userPoint >= 0 && userPoint < cropSize)
        userPointValid = false;

    UserPoint newUserPoint(userPointValid, userPoint, kit.userPoint.signal, kit.userPoint.signalAlt);
    UserPointKit newUserPointKit(newUserPoint, 0);

    ////

    AtOverlayMonitor atOverlayMonitor(kit.atVideoOverlay);
    AtVideoOverlayKit atOverlayKit(atOverlayMonitor, 0);

    //
    // Main process
    //

    require(processTarget(target, croppedFrame, frameIndex, stdPassKit(kitReplace(kit, kitCombine(newUserPointKit, atOverlayKit)))));

    //
    // If overlay is not set, use the cropped video image
    //

    if (kit.outputLevel >= OUTPUT_ENABLED && !atOverlayMonitor.overlayIsSet)
    {
        if (kit.dataProcessing)
        {
            AtProviderFromGpuImage imageProvider(croppedFrame, kit);
            require(kit.atVideoOverlay.setImage(croppedFrame.size(), imageProvider, STR("Cropped Frame"), 0, true, stdPass));
        }
    }

    ////

    stdEnd;
}

//================================================================
//
// VideoPreprocessorImpl::processEntry
//
//================================================================

bool VideoPreprocessorImpl::processEntry(VideoPrepTarget& target, stdPars(ProcessKit))
{
    stdBeginScoped;

    Matrix<const uint8_x4> cpuFrame = kit.atVideoFrame;

    ////

    if (displayFrameSize)
        printMsgL(kit, STR("Frame Size %0"), cpuFrame.size());

    ////

    if (alternativeVersion)
        printMsgL(kit, STR("Alternative Version!"), msgWarn);

    //----------------------------------------------------------------
    //
    // Normal execution: counting rollback = 0, execution rollback = 1
    // Repeated frame: counting rollback = 1, execution rollback = 1
    //
    // If video preprocessor has its own advancing process, we need to fix
    // repeated frame: counting rollback = 0, execution rollback = 1
    //
    //----------------------------------------------------------------

    bool internalProcess = randomizeSignal != 0;

    PipeControl pipeControl = kit.pipeControl;
    ProcessKit savedPipeKit = kit;

    REQUIRE(pipeControl.rollbackFrames == 0 || pipeControl.rollbackFrames == 1);

    if (internalProcess && !kit.dataProcessing)
        pipeControl.rollbackFrames = 0;

    ProcessKit kit = kitReplace(savedPipeKit, PipeControlKit(pipeControl, 0));

    ////

    if (randomizeSignal && !kit.dataProcessing)
        movingFrameIndex += 1;

    //----------------------------------------------------------------
    //
    // Add the video frame to the queue
    //
    //----------------------------------------------------------------

    GpuCopyThunk cpuFrameSender;

    ////

    if_not (!kit.frameAdvance && frameHistory.size() >= 1)
    {
        frameHistory.rollback(kit.pipeControl.rollbackFrames);

        FrameSnapshot* f = frameHistory.add();

        if (cpuFrame.memPitch() >= 0)
        {
            require(copyMatrixAsArray(cpuFrame, f->frameMemory, cpuFrameSender, stdPass));
            f->frame = f->frameMemory;
        }
        else
        {
            require(copyMatrixAsArray(flipMatrix(cpuFrame), f->frameMemory, cpuFrameSender, stdPass));
            f->frame = flipMatrix(f->frameMemory);
        }
    }

    //----------------------------------------------------------------
    //
    // Core processing
    //
    //----------------------------------------------------------------

    ProcessCheck processCheck;
    target.inspectProcess(processCheck);

    bool allSteady = processCheck.allSteady && prepParamsSteady && !randomizeSignal;

    ////

    if (kit.dataProcessing)
    {
        ProcessReset inspector;
        target.inspectProcess(inspector); // clear steady state
        prepParamsSteady = true;
    }

    ////

    REQUIRE(frameHistory.size() != 0);

    //
    // Re-feed
    //

    {
        TraceLocation processLocation = TRACE_AUTO_LOCATION;

        if (allSteady)
        {
            FrameSnapshot* f = frameHistory[0];
            REQUIRE(f != 0);

            require(processSingleFrame(target, f->frame, movingFrameIndex, stdPassKitLocation(kit, processLocation)));
        }
        else
        {
            Space histSize = frameHistory.size();
            REQUIRE(histSize >= 1);

            // target advance: -rollbackFrames + 1
            // initial rollback then feed: -initialRollback + histSize
            // initialRollback = histSize - 1 + rollbackFrames

            Space initialRollback = histSize - 1 + kit.pipeControl.rollbackFrames;

            if (histSize != 1)
                printMsgL(kit, STR("Re-feeding %0 frames!"), histSize, msgErr);

            for (Space k = histSize-1; k >= 0; --k)
            {
                FrameSnapshot* f = frameHistory[k];
                REQUIRE(f != 0);

                bool randomize = (randomizeSignal != 0) && !kit.dataProcessing;
                PipeControl pipeControl = (k == histSize-1) ? PipeControl(initialRollback, randomize) : PipeControl(0, false);
                PipeControlKit pipelineControlKit(pipeControl, 0);

                OutputLevelKit outputKit(k == 0 ? kit.outputLevel : OUTPUT_NONE);
                ProfilerKit profilerKit(k == 0 ? kit.profiler : 0);

                require
                (
                    processSingleFrame
                    (
                        target,
                        f->frame,
                        movingFrameIndex - k,
                        stdPassKitLocation(kitReplace(kit, kitCombine(pipelineControlKit, outputKit, profilerKit)), processLocation)
                    )
                );
            }
        }
    }

    ////

    stdEndScoped;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(VideoPreprocessor)
CLASSTHUNK_VOID1(VideoPreprocessor, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(VideoPreprocessor, reallocValid)
CLASSTHUNK_BOOL_STD1(VideoPreprocessor, realloc, const Point<Space>&, ReallocKit)
CLASSTHUNK_PURE0(VideoPreprocessor, Point<Space>, point(0), outputFrameSize, const)
CLASSTHUNK_BOOL_STD1(VideoPreprocessor, processEntry, VideoPrepTarget&, ProcessKit)

//----------------------------------------------------------------

}