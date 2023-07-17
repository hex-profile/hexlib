#if HOSTCODE
#include "gpuConsoleDrawing.h"
#endif

#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/gpuTool.h"
#include "imageRead/positionTools.h"
#include "gui/gpuConsoleDrawing/fontMono.h"
#include "gui/gpuConsoleDrawing/gpuConsoleDrawingTypes.h"

#if HOSTCODE
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "dataAlloc/arrayMemory.inl"
#include "dataAlloc/matrixMemory.inl"
#include "dataAlloc/gpuMatrixMemory.inl"
#include "copyMatrixAsArray.h"
#include "userOutput/printMsgEx.h"
#include "storage/rememberCleanup.h"
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
    ((Point<Space>, org))
    ((Point<float32>, divCharSize))
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

    auto charOfs = dstIdx - (textIdx * font.charSize);

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

stdbool GpuFontMonoMemory::realloc(const CpuFontMono& font, stdPars(GpuModuleReallocKit))
{
    REQUIRE(fontValid(font));

    allocated = false;
    allocFont = fontMonoNull<GpuPtr(const FontElement)>();

    ////

    require(gpuFontData.realloc(font.data.size(), stdPass));

    ////

    GpuCopyThunk gpuCopy; // wait on function exit
    require(gpuCopy(font.data, gpuFontData, stdPass));

    ////

    allocFont = fontMono(makeConst(gpuFontData()), font.charSize, font.dotValue, font.rangeOrg, font.rangeSize);
    REQUIRE(fontValid(allocFont));

    allocated = true;

    returnTrue;
}

//================================================================
//
// GpuConsoleDrawer::serialize
//
//================================================================

void GpuConsoleDrawer::serialize(const CfgSerializeKit& kit)
{
    displayWaitWarning.serialize(kit, STR("Display Wait Warning"));
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

stdbool GpuConsoleDrawer::realloc(const Point<Space>& textBufferSize, stdPars(ReallocKit))
{
    REQUIRE(textBufferSize >= 0);

    ////

    dealloc();

    ////

    require(copyQueue.realloc(copyQueueMaxSize, stdPass));

    ////

    for_count (i, copyQueueMaxSize)
    {
        CopyQueueSnapshot* s = copyQueue.add();
        REQUIRE(s != 0);

        require(s->cpuTextBuffer.realloc(textBufferSize, kit.gpuProperties.samplerAndFastTransferBaseAlignment, 1, kit.cpuFastAlloc, stdPass));
        require(s->gpuTextBuffer.reallocEx(textBufferSize, kit.gpuProperties.samplerAndFastTransferBaseAlignment, 1, kit.gpuFastAlloc, stdPass));

        if (kit.dataProcessing)
            require(kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, s->copyFinishEvent, stdPass));
    }

    ////

    allocTextBufferSize = textBufferSize;
    allocated = true;

    returnTrue;
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

stdbool GpuConsoleDrawer::drawText
(
    ColorTextProvider& buffer,
    const GpuMatrix<uint8_x4>& destination,
    const Point<Space>& renderOrg,
    const Point<Space>& renderEnd,
    const GpuFontMono& font,
    OptionalObject<RectRange>* actualRange,
    stdPars(ProcessKit)
)
{
    REQUIRE(allocated);

    ////

    REQUIRE(point(0) <= renderOrg && renderOrg <= renderEnd && renderEnd <= destination.size());

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
        require(kit.gpuEventWaiting.waitEvent(s.copyFinishEvent, realWait, stdPass));

    if (realWait && displayWaitWarning)
        printMsgL(kit, STR("GPU text drawer: Real wait happened!"), msgWarn);

    //----------------------------------------------------------------
    //
    // How many full chars we can print?
    //
    //----------------------------------------------------------------

    REQUIRE(font.charSize >= 1);

    auto renderSize = renderEnd - renderOrg;
    REQUIRE(renderSize >= 0);

    auto usedBufferSize = renderSize / font.charSize;

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
        require(buffer(usedBufferSize.Y, receiver, stdPass));
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

    require(copyMatrixAsArray(cpuActualBuffer, gpuActualBuffer, textCopier, stdPass));

    if (kit.dataProcessing)
        require(kit.gpuEventRecording.recordEvent(s.copyFinishEvent, kit.gpuCurrentStream, stdPass));

    textCopier.cancelSync(); // Don't flush pipeline on exit!

    //----------------------------------------------------------------
    //
    // Render text
    //
    //----------------------------------------------------------------

    auto consoleStaticSize = usedBufferSize * font.charSize; // Known at counting stage

    ////

    auto consoleDynamicSize = clampMax(gpuActualBuffer.size() * font.charSize, renderSize);

    ////

    GPU_MATRIX_ALLOC(tmpMatrix, uint8_x4, consoleStaticSize);
    REQUIRE(tmpMatrix.resize(consoleDynamicSize));

    ////

    require(renderText(recastElement<uint32>(tmpMatrix), gpuActualBuffer, font, renderOrg, 1 / convertFloat32(font.charSize), stdPass));

    ////

    auto consoleOrg = renderOrg;
    auto consoleEnd = renderOrg + consoleDynamicSize;

    if (actualRange && kit.dataProcessing)
        *actualRange = RectRange{consoleOrg, consoleEnd};

    auto targetOrg = clampRange(consoleOrg - 1, 0, destination.size());
    auto targetEnd = clampRange(consoleEnd + 1, 0, destination.size());

    GpuMatrix<uint8_x4> dstMatrix;
    REQUIRE(destination.subr(targetOrg, targetEnd, dstMatrix));

    if (hasData(tmpMatrix))
        require(textDilateBorder(tmpMatrix, dstMatrix, convertFloat32(targetOrg - consoleOrg), zeroOf<uint8_x4>(), stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

#endif

}

