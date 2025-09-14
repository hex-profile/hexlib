#pragma once

#include "yuvLayoutConvert/cpuPlanarYuv.h"
#include "dataAlloc/arrayMemory.h"

//================================================================
//
// CpuPlanarYuv420
//
//================================================================

template <typename Element>
class CpuPlanarYuvMemory : public CpuPlanarYuv<Element>
{

public:

    template <typename Kit>
    inline void realloc(const Point<Space>& size, Space baseByteAlignment, AllocatorInterface<CpuAddrU>& allocator, stdPars(Kit))
    {
        this->data.assignNull();
        this->size = point(0);

        ////

        REQUIRE(yuv420SizeValid(size));
        Space totalSize = yuv420TotalArea(size);

        allocData.realloc(totalSize, baseByteAlignment, allocator, stdPass);

        ////

        this->data = allocData;
        this->size = size;
    }

public:

    ArrayMemory<Element> allocData;

};
