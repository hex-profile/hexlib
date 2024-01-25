#pragma once

#include "cfgTools/numericVar.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "dataAlloc/matrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuLayer/gpuLayer.h"
#include "gpuModuleKit.h"
#include "gpuProcessHeader.h"
#include "history/historyObject.h"
#include "gui/gpuConsoleDrawing/fontMono.h"
#include "gui/gpuConsoleDrawing/gpuConsoleDrawingTypes.h"
#include "storage/adapters/callable.h"
#include "gui/common/rectRange.h"

namespace gpuConsoleDrawing {

//================================================================
//
// ColorTextReceiver
//
// Adds a row of text.
//
//================================================================

using ColorTextReceiver = Callable<void (const CharArray& text, uint32 color)>;

//================================================================
//
// ColorTextProvider
//
// Gets the last "max count" lines.
//
//================================================================

using ColorTextProvider = Callable<stdbool (Space maxCount, ColorTextReceiver& receiver, stdParsNull)>;

//================================================================
//
// GpuFontMonoMemory
//
//================================================================

class GpuFontMonoMemory
{

public:

    GpuFontMonoMemory();
    ~GpuFontMonoMemory();

public:

    inline operator GpuFontMono() const
        {return allocFont;}

public:

    stdbool realloc(const CpuFontMono& font, stdPars(GpuModuleReallocKit));

private:

    bool allocated = false;
    GpuArrayMemory<FontElement> gpuFontData;
    GpuFontMono allocFont;

};

//================================================================
//
// CopyQueueSnapshot
//
//================================================================

struct CopyQueueSnapshot
{
    GpuEventOwner copyFinishEvent;
    MatrixMemory<ConsoleElement> cpuTextBuffer;
    GpuMatrixMemory<ConsoleElement> gpuTextBuffer;
};

//================================================================
//
// PadMode
//
//================================================================

enum class PadMode {None, UsedSpace, EntireSpace, COUNT};

//================================================================
//
// DrawText
//
//================================================================

struct DrawText
{
    ColorTextProvider* buffer{};
    GpuMatrix<uint8_x4> destination;
    Point<Space> renderOrg{};
    Point<Space> renderEnd{};
    Point<Space> border = point(0);
    const GpuFontMono* font{};
    Space fontUpscalingFactor = 1;

    PadMode padMode = PadMode::None;
    Point3D<float32> padColor{};
    float32 padOpacity = 1;

    Point3D<float32> outlineColor = point3D(0.f);
};

//================================================================
//
// GpuConsoleDrawer
//
//================================================================

class GpuConsoleDrawer
{

public:

    void serialize(const CfgSerializeKit& kit);

public:

    using ReallocKit = KitCombine<CpuFuncKit, GpuAppFullKit>;

    stdbool realloc(const Point<Space>& textBufferSize, stdPars(ReallocKit));

    void dealloc();

public:

    using ProcessKit = KitCombine<GpuProcessKit, LocalLogKit>;

    stdbool drawText(const DrawText& args, stdPars(ProcessKit));

private:

    bool allocated = false;
    Point<Space> allocTextBufferSize = point(0);

    static const Space copyQueueMaxSize = 8;
    using CopyQueue = HistoryObjectStatic<CopyQueueSnapshot, copyQueueMaxSize>;
    CopyQueue copyQueue;

    BoolVar displayWaitWarning{true};

};

//----------------------------------------------------------------

}

using gpuConsoleDrawing::GpuFontMonoMemory;
using gpuConsoleDrawing::GpuConsoleDrawer;
