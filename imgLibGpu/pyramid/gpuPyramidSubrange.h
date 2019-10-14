#pragma once

#include "errorLog/errorLog.h"
#include "pyramid/gpuPyramid.h"
#include "numbers/divRound.h"

//================================================================
//
// GpuPyramidSubrange
//
//================================================================

template <typename Type>
class GpuPyramidSubrange : public GpuPyramid<Type>
{

public:

    virtual Space levels() const
    {
        Space reducedLevels = clampMin(basePyramid->levels() - startLevel, 0);
        if (decimationFactor != 1) reducedLevels = divUpNonneg(reducedLevels, decimationFactor);
        return clampMax(reducedLevels, maxLevels);
    }

    virtual Space layers() const
        {return basePyramid->layers();}

    virtual Point<Space> levelSize(Space level) const
        {return basePyramid->levelSize(startLevel + level * decimationFactor);}

    virtual GpuMatrix<Type> operator[] (Space level) const
        {return basePyramid->operator [](startLevel + level * decimationFactor);}

    virtual GpuMatrix<Type> getLayer(Space level, Space layer) const
        {return basePyramid->getLayer(startLevel + level * decimationFactor, layer);}

    virtual const GpuLayeredMatrix<Type>& getLevel(Space level) const
        {return basePyramid->getLevel(startLevel + level * decimationFactor);}

public:

    virtual bool getGpuLayout(GpuPtr(Type)& basePointer, GpuPyramidLayout& layout) const
    {
        GpuPyramidLayout baseLayout;
        ensure(basePyramid->getGpuLayout(basePointer, baseLayout));

        ////

        Space reducedLevels = clampMin(basePyramid->levels() - startLevel, 0);
        if (decimationFactor != 1) reducedLevels = divUpNonneg(reducedLevels, decimationFactor);
        Space levels = clampMax(reducedLevels, maxLevels);

        ////

        layout.levels = levels;
        layout.layers = baseLayout.layers;

        for (Space level = 0; level < levels; ++level)
        {
            Space baseLevel = startLevel + level * decimationFactor;
            layout.levelData[level] = baseLayout.levelData[baseLevel];
        }

        return true;
    }

public:

    inline void clear()
    {
        resetObject(*this);
    }

    inline void setupRange(GpuPyramid<Type>& basePyramid, Space startLevel = 0, Space maxLevels = spaceMax, Space decimationFactor = 1)
    {
        this->basePyramid = &basePyramid;
        this->startLevel = clampMin(startLevel, 0);
        this->maxLevels = clampMin(maxLevels, 0);
        this->decimationFactor = clampMin(decimationFactor, 1);
    }

private:

    GpuPyramid<Type>* basePyramid = nullptr;
    Space startLevel = 0;
    Space maxLevels = spaceMax;
    Space decimationFactor = 1;

};
