#pragma once

#include "data/matrix.h"
#include "yuvLayoutConvert/yuv420Tools.h"

//================================================================
//
// CpuPlanarYuv
//
// YUV image in 4:2:0 format, as sequential uint8 array.
//
//================================================================

template <typename Type>
struct CpuPlanarYuv
{
    Array<Type> data;
    Point<Space> size;

    inline CpuPlanarYuv()
        : size(point(0)) {}

    inline CpuPlanarYuv(const Array<Type>& data, const Point<Space>& size)
        : data(data), size(size)
    {
        if_not (yuv420TotalArea(size) == data.size())
            this->size = point(0);
    }

    inline void assignNull()
    {
        data.assignNull();
        size = point(0);
    }

    inline operator CpuPlanarYuv<const Type> () const
        {return CpuPlanarYuv<const Type>(data, size);}
};
