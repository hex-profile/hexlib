#if HOSTCODE
#include "gpuConsoleDrawing.h"
#endif

#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/gpuTool.h"
#include "gui/gpuConsoleDrawing/fontMono.h"
#include "gui/gpuConsoleDrawing/gpuConsoleDrawingTypes.h"
#include "imageRead/positionTools.h"
#include "numbers/mathIntrinsics.h"
#include "vectorTypes/vectorOperations.h"

#if HOSTCODE
#include "copyMatrixAsArray.h"
#include "dataAlloc/arrayMemory.inl"
#include "dataAlloc/gpuMatrixMemory.inl"
#include "dataAlloc/matrixMemory.inl"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsgEx.h"
#endif

namespace gpuConsoleDrawing {

//================================================================
//
// renderText
//
//================================================================

GPUTOOL_2D_BEG
(
    renderText,
    PREP_EMPTY,
    ((uint32, destination)),
    ((GpuMatrix<const ConsoleElement>, textBuffer))
    ((GpuFontMono, font))
    ((Point<float32>, divCharSize))
    ((Space, fontUpscalingFactor))
)
#if DEVCODE
{
    *destination = 0x00000000;

    ////

    Point<Space> dstIdx = point(X, Y);

    ////

    auto dstPos = point(Xs, Ys);

    auto textIdx = convertToNearestIndex(dstPos * divCharSize);
    ensurev(textBuffer.validAccess(textIdx));

    auto consoleValue = textBuffer.read(textIdx);

    ////

    auto charOfs = dstIdx - (textIdx * font.charSize * fontUpscalingFactor);

    if (fontUpscalingFactor != 1)
    {
        if (fontUpscalingFactor == 2)
            charOfs >>= 1;
        else
            charOfs /= fontUpscalingFactor;
    }

    ////

    auto charIndex = Space(consoleElementLetter(consoleValue)) - font.rangeOrg;
    ensurev(SpaceU(charIndex) < SpaceU(font.rangeSize));

    ////

    auto charImageArea = areaOf(font.charSize);

    auto ofs = charIndex * charImageArea + charOfs.Y * font.charSize.X + charOfs.X;
    ensurev(font.data.validAccess(ofs));

    auto value = font.data.read(ofs);
    ensurev(value == font.dotValue);

    *destination = consoleElementColor(consoleValue) | 0xFF000000;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// textDilateBorder
//
//================================================================

GPUTOOL_2D_BEG
(
    textDilateBorder,
    ((const uint8_x4, src, INTERP_NEAREST, BORDER_ZERO)),
    ((uint8_x4, dst)),
    ((Point<float32>, srcOfs))
    ((uint8_x4, shadowColor))
)
#if DEVCODE
{
    Point<float32> srcPos = point(Xs, Ys) + srcOfs;

    ////

    float32 maxAlpha = 0;

    devUnrollLoop
    for (Space dY = -1; dY <= +1; ++dY)
    {
        devUnrollLoop
        for (Space dX = -1; dX <= +1; ++dX)
        {
            if_not (dX == 0 && dY == 0)
            {
                auto value = tex2D(srcSampler, (srcPos + convertFloat32(point(dX, dY))) * srcTexstep);
                maxAlpha = maxv(maxAlpha, value.w);
            }
        }
    }

    ////

    auto centralValue = tex2D(srcSampler, srcPos * srcTexstep);

    bool textDot = (centralValue.w != 0);
    bool shadowDot = (maxAlpha != 0);

    uint8_x4 storedValue = shadowColor;
    if (textDot) storedValue = convertNormClamp<uint8_x4>(centralValue);

    if (shadowDot || textDot)
        *dst = storedValue;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// overlayPad
//
//================================================================

GPUTOOL_2D_BEG
(
    overlayPad,
    PREP_EMPTY,
    ((uint8_x4, dst)),
    ((float32_x4, padColor))
    ((float32, padOpacity))
)
#if DEVCODE
{
    auto srcValue = loadNorm(dst);
    auto dstValue = linearIf(padOpacity, padColor, srcValue);
    storeNorm(dst, dstValue);
}
#endif
GPUTOOL_2D_END

//----------------------------------------------------------------

#if HOSTCODE

//================================================================
//
// GpuFontMonoMemory::GpuFontMonoMemory
//
//================================================================

GpuFontMonoMemory::GpuFontMonoMemory()
{
    allocFont = fontMonoNull<GpuPtr(const FontElement)>();
}

GpuFontMonoMemory::~GpuFontMonoMemory()
{
}

//================================================================
//
// GpuFontMonoMemory::realloc
//
//================================================================

void GpuFontMonoMemory::realloc(const CpuFontMono& font, stdPars(GpuModuleReallocKit))
{
    REQUIRE(fontValid(font));

    allocated = false;
    allocFont = fontMonoNull<GpuPtr(const FontElement)>();

    ////

    gpuFontData.realloc(font.data.size(), stdPass);

    ////

    GpuCopyThunk gpuCopy; // wait on function exit
    gpuCopy(font.data, gpuFontData, stdPass);

    ////

    allocFont = fontMono(makeConst(gpuFontData()), font.charSize, font.dotValue, font.rangeOrg, font.rangeSize);
    REQUIRE(fontValid(allocFont));

    allocated = true;
}

//================================================================
//
// GpuConsoleDrawer::serialize
//
//================================================================

void GpuConsoleDrawer::serialize(const CfgSerializeKit& kit)
{
}

//================================================================
//
// GpuConsoleDrawer::dealloc
//
//================================================================

void GpuConsoleDrawer::dealloc()
{
    allocated = false;
    allocTextBufferSize = point(0);

    ////

    copyQueue.reallocStatic(0);
}

//================================================================
//
// GpuConsoleDrawer
//
//================================================================

void GpuConsoleDrawer::realloc(const Point<Space>& textBufferSize, stdPars(ReallocKit))
{
    REQUIRE(textBufferSize >= 0);

    ////

    dealloc();

    ////

    copyQueue.realloc(copyQueueMaxSize, stdPass);

    ////

    for_count (i, copyQueueMaxSize)
    {
        CopyQueueSnapshot* s = copyQueue.add();
        REQUIRE(s != 0);

        s->cpuTextBuffer.realloc(textBufferSize, kit.gpuProperties.samplerAndFastTransferBaseAlignment, 1, kit.cpuFastAlloc, stdPass);
        s->gpuTextBuffer.reallocEx(textBufferSize, kit.gpuProperties.samplerAndFastTransferBaseAlignment, 1, kit.gpuFastAlloc, stdPass);

        if (kit.dataProcessing)
            kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, s->copyFinishEvent, stdPass);
    }

    ////

    allocTextBufferSize = textBufferSize;
    allocated = true;
}

//================================================================
//
// ReceiveToCpuBuffer
//
//================================================================

class ReceiveToCpuBuffer : public ColorTextReceiver
{

public:

    ReceiveToCpuBuffer(const Matrix<ConsoleElement>& dst)
        : dst{dst}
    {
    }

public:

    void operator()(const CharArray& text, uint32 color)
    {
        MATRIX_EXPOSE(dst);

        if_not (dstRow < dstSizeY)
            return;

        auto rowPtr = MATRIX_POINTER(dst, 0, dstRow);
        dstRow++;

        ////

        Space textSize = Space(clampMax(text.size, size_t(dstSizeX)));

        auto dstPtr = rowPtr;

        ////

        for_count_ex (X, textSize, ++dstPtr)
        {
            auto letter = text.ptr[X];
            *dstPtr = consoleElementCompose(letter, color);
        }

        ////

        for (Space X = textSize; X < dstSizeX; ++X, ++dstPtr)
            *dstPtr = consoleElementCompose(' ', 0);

        ////

        if_not (text.size <= size_t(dstSizeX))
        {
            if (dstSizeX)
                rowPtr[dstSizeX-1] = consoleElementCompose('>', 0x00FF0000);
        }
    }

public:

    void finish()
    {
        MATRIX_EXPOSE(dst);
        auto rowPtr = MATRIX_POINTER(dst, 0, dstRow);

        for (Space Y = dstRow; Y < dstSizeY; ++Y, rowPtr += dstMemPitch)
        {
            auto dstPtr = rowPtr;

            for_count_ex (X, dstSizeX, ++dstPtr)
                *dstPtr = consoleElementCompose(' ', 0);
        }
    }

public:

    Space actualRowCount() const
        {return dstRow;}

private:

    Space dstRow = 0;
    Matrix<ConsoleElement> dst;

};

//================================================================
//
// GpuConsoleDrawer::drawText
//
//================================================================

void GpuConsoleDrawer::drawText(const DrawText& args, stdPars(ProcessKit))
{
    stdScopedBegin;

    REQUIRE(allocated);

    ////

    auto& destination = args.destination;

    REQUIRE(args.font);
    auto& font = *args.font;

    REQUIRE(args.buffer);
    auto& buffer = *args.buffer;

    auto& border = args.border;

    ////

    REQUIRE(args.fontUpscalingFactor >= 1);
    auto fontCharSize = font.charSize * args.fontUpscalingFactor;

    ////

    REQUIRE(point(0) <= args.renderOrg && args.renderOrg <= args.renderEnd && args.renderEnd <= destination.size());

    //----------------------------------------------------------------
    //
    // Apply border.
    //
    //----------------------------------------------------------------

    REQUIRE(border >= 0);

    auto renderOrg = args.renderOrg + border;
    auto renderEnd = args.renderEnd - border;

    if_not (renderOrg < renderEnd)
        return;

    //----------------------------------------------------------------
    //
    // Get a new queue snapshot.
    //
    //----------------------------------------------------------------

    HistoryState copyQueueState;
    copyQueue.saveState(copyQueueState);

    REMEMBER_COUNTING_CLEANUP(copyQueue.restoreState(copyQueueState));

    ////

    auto* snapshotPtr = copyQueue.add();
    REQUIRE(snapshotPtr != 0);
    auto& s = *snapshotPtr;

    //----------------------------------------------------------------
    //
    // Wait for the end of the previous copying from CPU text buffer to GPU.
    //
    //----------------------------------------------------------------

    bool realWait = false;

    if (kit.dataProcessing)
        kit.gpuEventWaiting.waitEvent(s.copyFinishEvent, realWait, stdPass);

    if (realWait && 1)
        printMsgL(kit, STR("GPU text drawer: Real wait happened!"), msgWarn);

    //----------------------------------------------------------------
    //
    // How many full chars we can print?
    //
    //----------------------------------------------------------------

    REQUIRE(fontCharSize >= 1);

    auto renderSize = renderEnd - renderOrg;
    REQUIRE(renderSize >= 0);

    auto usedBufferSize = renderSize / fontCharSize;

    usedBufferSize = clampMax(usedBufferSize, allocTextBufferSize);

    //----------------------------------------------------------------
    //
    // Copy to CPU text buffer
    //
    //----------------------------------------------------------------

    auto matrixIsDense = [] (auto& matrix) {return matrix.memPitch() == matrix.sizeX();};

    ////

    REQUIRE(s.cpuTextBuffer.resize(usedBufferSize));
    REQUIRE(matrixIsDense(s.cpuTextBuffer));

    REQUIRE(s.gpuTextBuffer.resize(usedBufferSize));
    REQUIRE(matrixIsDense(s.gpuTextBuffer));

    ////

    Space actualRowCount = 0;

    if (kit.dataProcessing)
    {
        ReceiveToCpuBuffer receiver{s.cpuTextBuffer};
        buffer(usedBufferSize.Y, receiver, stdPass);
        receiver.finish();
        actualRowCount = receiver.actualRowCount();
    }

    //----------------------------------------------------------------
    //
    // Copy CPU text buffer to GPU
    //
    //----------------------------------------------------------------

    Matrix<ConsoleElement> cpuActualBuffer;
    REQUIRE(s.cpuTextBuffer.subs(0, 0, s.cpuTextBuffer.sizeX(), actualRowCount, cpuActualBuffer));

    GpuMatrix<ConsoleElement> gpuActualBuffer;
    REQUIRE(s.gpuTextBuffer.subs(0, 0, s.cpuTextBuffer.sizeX(), actualRowCount, gpuActualBuffer));

    ////

    GpuCopyThunk textCopier;

    copyMatrixAsArray(cpuActualBuffer, gpuActualBuffer, textCopier, stdPass);

    if (kit.dataProcessing)
        kit.gpuEventRecording.recordEvent(s.copyFinishEvent, kit.gpuCurrentStream, stdPass);

    textCopier.cancelSync(); // Don't flush pipeline on exit!

    ////

    auto consoleStaticSize = usedBufferSize * fontCharSize; // Known at counting stage

    ////

    auto consoleDynamicSize = clampMax(gpuActualBuffer.size() * fontCharSize, renderSize);

    //----------------------------------------------------------------
    //
    // Render pad if any.
    //
    //----------------------------------------------------------------

    if (args.padMode != PadMode::None && allv(consoleDynamicSize > 0))
    {
        auto padOrg = args.renderOrg;
        auto padEnd = args.renderEnd;

        if (args.padMode == PadMode::UsedSpace)
        {
            padEnd.Y = renderOrg.Y + consoleDynamicSize.Y;

            padOrg = clampRange(padOrg - border, args.renderOrg, args.renderEnd);
            padEnd = clampRange(padEnd + border, args.renderOrg, args.renderEnd);
        }

        auto color = convertNearest<float32_x4>(args.padColor);

        GpuMatrix<uint8_x4> padArea;
        REQUIRE(destination.subr(padOrg, padEnd, padArea));

        if (args.padOpacity == 1)
        {
            gpuMatrixSet(padArea, convertNormClamp<uint8_x4>(color), stdPass);
        }
        else
        {
            overlayPad(padArea, color, args.padOpacity, stdPass);
        }
    }

    //----------------------------------------------------------------
    //
    // Render text
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(tmpMatrix, uint8_x4, consoleStaticSize);
    REQUIRE(tmpMatrix.resize(consoleDynamicSize));

    ////

    renderText(recastElement<uint32>(tmpMatrix), gpuActualBuffer, font,
        1 / convertFloat32(fontCharSize), args.fontUpscalingFactor, stdPass);

    ////

    auto consoleOrg = renderOrg;
    auto consoleEnd = renderOrg + consoleDynamicSize;

    auto targetOrg = clampRange(consoleOrg - 1, 0, destination.size());
    auto targetEnd = clampRange(consoleEnd + 1, 0, destination.size());

    GpuMatrix<uint8_x4> dstMatrix;

    REQUIRE(destination.subr(targetOrg, targetEnd, dstMatrix));

    if (hasData(tmpMatrix))
    {
        auto outlineColor = convertNearest<float32_x4>(args.outlineColor);

        textDilateBorder(tmpMatrix, dstMatrix, convertFloat32(targetOrg - consoleOrg), convertNormClamp<uint8_x4>(outlineColor), stdPass);
    }

    ////

    stdScopedEnd;
}

//----------------------------------------------------------------

#endif

}

