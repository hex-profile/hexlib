#include "videoPreprocessor.h"

#define USE_OVERLAY_SMOOTHER 0

#if USE_OVERLAY_SMOOTHER
    #include "atAssembly/videoPreprocessor/overlaySmoother.h"
#endif

#include "atAssembly/floatRangesTest/floatRangesTest.h"
#include "gpuBaseConsoleByCpu/gpuBaseConsoleByCpu.h"
#include "atAssembly/halfFloatTest/halfFloatTest.h"
#include "atAssembly/videoPreprocessor/displayWaitController.h"
#include "atAssembly/videoPreprocessor/tools/rotateImage.h"
#include "atAssembly/videoPreprocessor/tools/videoPrepTools.h"
#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "cfgTools/numericVar.h"
#include "cfgTools/rangeValueControl.h"
#include "configFile/cfgSimpleString.h"
#include "copyMatrixAsArray.h"
#include "dataAlloc/arrayMemory.inl"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "flipMatrix.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuImageConsoleImpl/gpuImageConsoleImpl.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "history/historyObject.h"
#include "kits/displayParamsKit.h"
#include "kits/userPoint.h"
#include "mathFuncs/rotationMath.h"
#include "baseConsoleAvi/baseConsoleAvi.h"
#include "randomImage/randomImage.h"
#include "rndgen/rndgenFloat.h"
#include "storage/classThunks.h"
#include "storage/rememberCleanup.h"
#include "timer/timer.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsgEx.h"
#include "displayParamsImpl/displayParamsImpl.h"
#include "atInterface/atInterface.h"

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

    stdbool setImage(const Point<Space>& size, AtImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
        {overlayIsSet = true; return base.setImage(size, imageProvider, desc, id, textEnabled, stdNullPassThru);}

    stdbool setImageFake(stdNullPars)
        {overlayIsSet = true; return base.setImageFake(stdNullPassThru);}

    stdbool updateImage(stdNullPars)
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
    SimpleStringVar outputDir{STR("")};
    SimpleStringVar outputCodec{STR("DIB ")};
    NumericVarStatic<int32, 0, 0x7FFFFFFF, 256> maxSegmentFrames;

public:

    void serialize(const ModuleSerializeKit& kit)
    {
        savingActive.serialize(kit, STR("Active"), STR("Shift+Alt+V"));
        outputFps.serialize(kit, STR("Playback FPS"), STR("Playback framerate specified in AVI header"));

        outputDir.serialize(kit, outputDirName(), STR("Use double backslashes, for example C:\\\\Temp"));
        outputCodec.serialize(kit, STR("Compressor FourCC"), STR("Use 'DIB ' for uncompressed, 'ffds' for ffdshow"));

        maxSegmentFrames.serialize(kit, STR("Max Video Segment Frames"));
    }

};

//================================================================
//
// ProcessExKit
//
//================================================================

KIT_COMBINE4(ProcessExKit, ProcessKit, GpuImageConsoleKit, AlternativeVersionKit, DisplayParamsKit);

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

    void serialize(const ModuleSerializeKit& kit);
    void setFrameSize(const Point<Space>& frameSize);

    bool reallocValid() const;
    stdbool realloc(stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;

    stdbool process(VideoPrepTarget& target, stdPars(ProcessKit));

    stdbool processSingleFrame
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    stdbool processPrepFrontend
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    stdbool processCropFrontend
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

    stdbool processTarget
    (
        VideoPrepTarget& target,
        const GpuMatrix<const uint8_x4>& inputFrame,
        uint32 frameIndex,
        stdPars(ProcessKit)
    );

private:

    Point<Space> desiredFrameSize = point(0);
    Point<Space> allocFrameSize = point(0);

    ////

    NumericVarStatic<Space, 1, maxFrameHistoryCapacity, 1> frameHistoryCapacity;
    using FrameHistory = HistoryObjectStatic<FrameSnapshot, maxFrameHistoryCapacity>;
    FrameHistory frameHistory;

private:

    BoolSwitch<false> displayFrameSize;

    ////

    bool prepParamsSteady = true;

    ////

    BoolSwitch<false> cropMode;
    NumericVar<Point<Space>> cropSizeCfg{point(0), point(8192), point(1280, 720)};

    ////

    enum GenMode {GenNone, GenPulse, GenGrating, GenRandom, GenModeCount};
    RingSwitch<GenMode, GenModeCount, GenNone> genMode;

    RangeValueControl<float32> genGratingPeriod{2, 2048, 6, 1.02189714865411668f, RangeValueLogscale};
    BoolSwitch<false> genGratingRectangleShape;

    NumericVarStatic<Space, 1, 1 << 20, 256> genPulsePeriod;

    ////

    RangeValueControl<float32> rotationAngle{0, 1, 0, 1.f/128, RangeValueCircular};

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
    NumericVar<float32> noiseSigma{0, 1, 0.01f};

private:

    halfFloatTest::HalfFloatTest halfFloatTest;
    floatRangesTest::FloatRangesTest floatRangesTest;

private:

    DisplayWaitController displayWaitController;

    enum DisplayType {DisplayNothing, DisplayFrameHistory, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0xB0C09C28> displaySwitch;

private:

    DisplayParamsImpl displayParams;

private:

#if USE_OVERLAY_SMOOTHER
    BoolSwitch<true> overlaySmootherEnabled;
    bool overlaySmootherTried = false;
    bool overlaySmootherInit = false;
    overlaySmoother::OverlaySmoother overlaySmoother;
#endif

private:

    AviOutputConfig aviConfig;
    BaseConsoleAvi baseConsoleAvi;

private:

    bool rndgenFrameInitialized = false;
    GpuMatrixMemory<RndgenState> rndgenFrame;

};

//================================================================
//
// VideoPreprocessorImpl::serialize
//
//================================================================

void VideoPreprocessorImpl::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Display Params");

        displayParams.serialize(kit, prepParamsSteady);
    }

    {
        CFG_NAMESPACE("Input Settings");

        displaySwitch.serialize
        (
            kit, STR("Display"),
            {STR("<Nothing>"), STR("")},
            {STR("Frame History"), STR("Shift+F")}
        );

        frameHistoryCapacity.serialize(kit, STR("Frame History Size"));
        displayFrameSize.serialize(kit, STR("Display Frame Size"), STR(""));

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
        CFG_NAMESPACE("Output Settings");

        {
            CFG_NAMESPACE("Display Delayer");
            displayWaitController.serialize(kit);
        }
    }

    {
    #if USE_OVERLAY_SMOOTHER

        CFG_NAMESPACE("Overlay Smoother");
        overlaySmootherEnabled.serialize(kit, STR("@Enabled"), STR(""));
        overlaySmoother.serialize(kit);

    #endif
    }

    {
        CFG_NAMESPACE("Saving AVI Files");
        aviConfig.serialize(kit);
    }

    {
        CFG_NAMESPACE("~Tests");
        halfFloatTest.serialize(kit);
        floatRangesTest.serialize(kit);
    }
}

//================================================================
//
// VideoPreprocessorImpl::setFrameSize
//
//================================================================

void VideoPreprocessorImpl::setFrameSize(const Point<Space>& frameSize)
{
    desiredFrameSize = frameSize;
}

//================================================================
//
// VideoPreprocessorImpl::reallocValid
//
//================================================================

bool VideoPreprocessorImpl::reallocValid() const
{
    return
        allv(allocFrameSize == desiredFrameSize) &&
        frameHistoryCapacity == frameHistory.allocSize();
}

//================================================================
//
// VideoPreprocessorImpl::realloc
//
//================================================================

stdbool VideoPreprocessorImpl::realloc(stdPars(ReallocKit))
{
    allocFrameSize = point(0);

    ////

    require(frameHistory.realloc(frameHistoryCapacity, stdPass));

    ////

    for (Space k = 0; k < frameHistoryCapacity; ++k)
    {
        FrameSnapshot* f = frameHistory.add();
        require(f->frameMemory.realloc(desiredFrameSize, stdPass));
        f->frame = f->frameMemory;
    }

    frameHistory.clear();
    movingFrameIndex = 0;

    ////

    require(rndgenFrame.realloc(desiredFrameSize, stdPass));
    rndgenFrameInitialized = false;

    ////

    allocFrameSize = desiredFrameSize;

    returnTrue;
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

stdbool VideoPreprocessorImpl::processTarget
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdScopedBegin;

    //----------------------------------------------------------------
    //
    // Set AVI writer parameters
    //
    //----------------------------------------------------------------

    auto setAviOutput = [&] () -> stdbool
    {
        if_not (aviConfig.outputDir->length() != 0)
        {
            printMsgL(kit, STR("AVI Saving: <%0> is not set (Testbed->Config->Edit)"), aviConfig.outputDirName(), msgWarn);
            returnFalse;
        }

        require(baseConsoleAvi.setOutputDir(aviConfig.outputDir->cstr(), stdPass));
        require(baseConsoleAvi.setCodec(baseConsoleAvi::codecFromStr(aviConfig.outputCodec->cstr()), stdPass));
        require(baseConsoleAvi.setFps(aviConfig.outputFps, stdPass));
        require(baseConsoleAvi.setMaxSegmentFrames(aviConfig.maxSegmentFrames, stdPass));

        returnTrue;
    };

    ////

    bool aviOk = false;
    
    if (aviConfig.savingActive)
    {
        aviOk = errorBlock(setAviOutput());

        if (aviOk)
            printMsgL(kit, STR("AVI Saving: Files are saved to %0 (playback %1 fps, compressor '%2')"),
                aviConfig.outputDir->cstr(), aviConfig.outputFps(), aviConfig.outputCodec->cstr());
        else
            printMsgL(kit, STR("AVI Saving: Error happened"), msgWarn);
    }

    ////

    AtImgConsole* atImageConsole = &kit.atImgConsole;
    AtVideoOverlay* atVideoOverlay = &kit.atVideoOverlay;

    ////

    BaseConsoleAviThunk baseConsoleAviThunk(baseConsoleAvi, *atImageConsole, *atVideoOverlay, kit);

    if (aviOk)
        {atImageConsole = &baseConsoleAviThunk; atVideoOverlay = &baseConsoleAviThunk;}

    //----------------------------------------------------------------
    //
    // GPU image console.
    // Display params.
    //
    //----------------------------------------------------------------

    GpuBaseConsoleProhibitThunk gpuBaseConsoleProhibited(kit);
    GpuBaseConsoleByCpuThunk gpuBaseConsoleAt(*atImageConsole, *atVideoOverlay, kit);
    GpuBaseConsole* gpuBaseConsole = &gpuBaseConsoleProhibited;

    if (kit.verbosity >= Verbosity::On)
        gpuBaseConsole = &gpuBaseConsoleAt;

    auto& dp = displayParams;

    GpuImageConsoleThunk gpuImageConsole(*gpuBaseConsole, dp.displayMode(), dp.vectorMode(), kit);

    ////

    DisplayParamsThunk displayParamsThunk{inputFrame.size(), displayParams};

    ////

    auto oldKit = kit;
    auto kit = kitCombine(oldKit, GpuImageConsoleKit(gpuImageConsole), displayParamsThunk.getKit());

    //----------------------------------------------------------------
    //
    // Display frame history
    //
    //----------------------------------------------------------------

    DisplayType displayType = kit.verbosity >= Verbosity::Render ? displaySwitch : DisplayNothing;

    if (displayType == DisplayFrameHistory)
    {
        if (frameHistory.size() == 0)
            printMsgL(kit, STR("Video Preprocessor: Frame history empty"), msgErr);
        else
        {
            Space i = kit.display.temporalIndex(-(frameHistory.size()-1), 0);

            require(kit.gpuImageConsole.addRgbColorImage(makeConst(frameHistory[-i]->frameMemory),
                0x00, 0xFF * kit.display.factor, point(1.f), INTERP_NEAREST, point(0), BORDER_ZERO,
                paramMsg(STR("Video Preprocessor: Frame history [%0]"), i), stdPass));
        }
    }

    //----------------------------------------------------------------
    //
    // Call target
    //
    //----------------------------------------------------------------

    DisplayDelayerThunk displayDelayer(displayWaitController, kit);

    require(target.process(stdPassKit(kitCombine(kit, GpuRgbFrameKit(inputFrame), DisplayDelayerKit(displayDelayer)))));

    ////

    stdScopedEnd;
}

//================================================================
//
// VideoPreprocessorImpl::processSingleFrame
//
//================================================================

stdbool VideoPreprocessorImpl::processSingleFrame
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    stdScopedBegin;

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

    auto kit = kitReplace(oldKit, AtVideoOverlayKit(overlaySmootherThunk));

    ////

    bool useOverlaySmoothing = overlaySmootherEnabled && (kit.atRunning && kit.atPlaying);

    require(overlaySmootherThunk.setSmoothing(useOverlaySmoothing, stdPass));

#endif

    //----------------------------------------------------------------
    //
    // Call
    //
    //----------------------------------------------------------------

    bool processOk = errorBlock(processPrepFrontend(target, inputFrame, frameIndex, stdPass));

    //----------------------------------------------------------------
    //
    // Overlay
    //
    //----------------------------------------------------------------

#if USE_OVERLAY_SMOOTHER

    if_not (overlaySmootherThunk.overlayIsSet)
    {
        if (kit.verbosity >= Verbosity::On)
        {
            GpuBaseImageProvider imageProvider(kit);
            require(imageProvider.setImage(inputFrame, stdPass));

            if (kit.dataProcessing)
                require(overlaySmootherThunk.setImage(inputFrame.size(), imageProvider, STR("Input Frame"), 0, true, stdPass));
        }
    }

    ////

    if_not (useOverlaySmoothing)
        require(overlaySmootherThunk.flushSmoothly(stdPass));

#endif

    ////

    require(processOk);

    stdScopedEnd;
}

//================================================================
//
// VideoPreprocessorImpl::processPrepFrontend
//
//================================================================

stdbool VideoPreprocessorImpl::processPrepFrontend
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    Point<Space> frameSize = inputFrame.size();

    AtOverlayMonitor atOverlayMonitor(kit.atVideoOverlay);
    ProcessKit kitMonitorEx = kitReplace(kit, AtVideoOverlayKit(atOverlayMonitor));

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
        returnTrue;
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

    Point<float32> forwardRotation = complexConjugate(circleCCW(rotationAngle()));

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
            usedRotation = complexConjugate(usedRotation);
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

    if (kit.verbosity >= Verbosity::On && !atOverlayMonitor.overlayIsSet)
    {
        GpuBaseImageProvider imageProvider(kit);
        require(imageProvider.setImage(processedFrame, stdPass));

        require(kit.atVideoOverlay.setImage(processedFrame.size(), imageProvider, STR("Rotated Frame"), 0, true, stdPass));
    }

    ////

    returnTrue;
}

//================================================================
//
// VideoPreprocessorImpl::processCropFrontend
//
//================================================================

stdbool VideoPreprocessorImpl::processCropFrontend
(
    VideoPrepTarget& target,
    const GpuMatrix<const uint8_x4>& inputFrame,
    uint32 frameIndex,
    stdPars(ProcessKit)
)
{
    Point<Space> frameSize = inputFrame.size();

    Point<Space> cropSize = clampRange(cropSizeCfg(), point(0), frameSize);

    if_not (cropMode)
        cropSize = frameSize;

    ////

    Point<Space> cropOfs = (frameSize - cropSize) >> 1;

    //
    // Input frame
    //

    GpuMatrix<const uint8_x4> croppedFrame = inputFrame;

    GPU_MATRIX_ALLOC(croppedFrameMemory, uint8_x4, cropSize);

    if_not (allv(cropSize == frameSize))
    {
        GpuMatrix<uint8_x4> dstFrame = flipMatrix(croppedFrameMemory);
        croppedFrame = dstFrame;
        require(copyImageRect(inputFrame, cropOfs, dstFrame, stdPass));
    }

    //
    // User point
    //

    bool userPointValid = kit.userPoint.valid;
    Point<Space> userPoint = kit.userPoint.position;

    if_not (userPoint >= 0 && userPoint < cropSize)
        userPointValid = false;

    UserPoint newUserPoint(userPointValid, userPoint, kit.userPoint.signal, kit.userPoint.signalAlt);
    UserPointKit newUserPointKit(newUserPoint);

    ////

    AtOverlayMonitor atOverlayMonitor(kit.atVideoOverlay);
    AtVideoOverlayKit atOverlayKit(atOverlayMonitor);

    //
    // Main process
    //

    require(processTarget(target, croppedFrame, frameIndex, stdPassKit(kitReplace(kit, kitCombine(newUserPointKit, atOverlayKit)))));

    //
    // If overlay is not set, use the cropped video image
    //

    if (kit.verbosity >= Verbosity::On && !atOverlayMonitor.overlayIsSet)
    {
        GpuBaseImageProvider imageProvider(kit);
        require(imageProvider.setImage(croppedFrame, stdPass));

        require(kit.atVideoOverlay.setImage(croppedFrame.size(), imageProvider, STR(""), 0, false, stdPass));
    }

    ////

    returnTrue;
}

//================================================================
//
// VideoPreprocessorImpl::process
//
//================================================================

stdbool VideoPreprocessorImpl::process(VideoPrepTarget& target, stdPars(ProcessKit))
{
    stdScopedBegin;

    Matrix<const uint8_x4> cpuFrame = kit.atVideoFrame;

    ////

    if (displayFrameSize)
        printMsgL(kit, STR("Frame Size %0"), cpuFrame.size());

    ////

    if (displayParams.alternativeVersion())
        printMsgL(kit, STR("Alternative Version!"), msgWarn);

    //----------------------------------------------------------------
    //
    // Tests.
    //
    //----------------------------------------------------------------

    if (kit.dataProcessing)
    {
        require(floatRangesTest.process(stdPass));
    }

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

    ProcessKit kit = kitReplace(savedPipeKit, PipeControlKit(pipeControl));

    ////

    if (randomizeSignal && !kit.dataProcessing)
        movingFrameIndex += 1;

    //----------------------------------------------------------------
    //
    // Add the video frame to the queue.
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
    // Core processing.
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

            require(processSingleFrame(target, f->frame, movingFrameIndex, stdPassEx(kit, processLocation)));
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
                PipeControlKit pipelineControlKit(pipeControl);

                VerbosityKit outputKit(k == 0 ? kit.verbosity : Verbosity::Off);
                ProfilerKit profilerKit(k == 0 ? kit.profiler : 0);

                require
                (
                    processSingleFrame
                    (
                        target,
                        f->frame,
                        movingFrameIndex - k,
                        stdPassEx(kitReplace(kit, kitCombine(pipelineControlKit, outputKit, profilerKit)), processLocation)
                    )
                );
            }
        }
    }

    ////

    stdScopedEnd;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(VideoPreprocessor)
CLASSTHUNK_VOID1(VideoPreprocessor, serialize, const ModuleSerializeKit&)
CLASSTHUNK_VOID1(VideoPreprocessor, setFrameSize, const Point<Space>&)
CLASSTHUNK_BOOL_CONST0(VideoPreprocessor, reallocValid)
CLASSTHUNK_BOOL_STD0(VideoPreprocessor, realloc, ReallocKit)
CLASSTHUNK_PURE0(VideoPreprocessor, Point<Space>, point(0), outputFrameSize, const)
CLASSTHUNK_BOOL_STD1(VideoPreprocessor, process, VideoPrepTarget&, ProcessKit)

//----------------------------------------------------------------

}
