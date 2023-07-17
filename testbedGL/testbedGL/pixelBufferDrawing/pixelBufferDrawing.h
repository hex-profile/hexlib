#pragma once

#include "allocation/mallocKit.h"
#include "storage/smartPtr.h"
#include "testbedGL/pixelBuffer/pixelBuffer.h"
#include "userOutput/diagnosticKit.h"

//================================================================
//
// PixelBufferDrawing
//
// Draws an BGRA memory buffer by OpenGL using pixel shader.
//
//================================================================

struct PixelBufferDrawing
{
    static UniquePtr<PixelBufferDrawing> create();
    virtual ~PixelBufferDrawing() {}

    ////

    virtual void deinit() =0;

    using ReinitKit = KitCombine<DiagnosticKit, MallocKit>;
    virtual stdbool reinit(stdPars(ReinitKit)) =0;

    ////

    virtual stdbool draw(const PixelBuffer& buffer, const Point<Space>& pos, stdPars(DiagnosticKit)) =0;
};
