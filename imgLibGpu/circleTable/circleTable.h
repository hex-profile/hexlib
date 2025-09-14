#pragma once

#include "dataAlloc/gpuMatrixMemory.h"
#include "gpuProcessHeader.h"

//================================================================
//
// CircleTableHolder
//
// Makes a table for using with texture sampler instead of sin/cos.
// The texture sample should be set to in BORDER_WRAP and INTERP_LINEAR.
//
//================================================================

class CircleTableHolder
{

public:

    void realloc(Space size, stdPars(GpuProcessKit));
    GpuMatrix<const float32_x2> operator () () const {return table;}

private:

    GpuMatrixMemory<float32_x2> table;

};
