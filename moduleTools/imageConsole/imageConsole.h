#pragma once

#include "numbers/int/intBase.h"
#include "point/point.h"
#include "data/space.h"
#include "charType/charArray.h"
#include "formatting/formatStream.h"
#include "formatting/formatOutputAtom.h"

//================================================================
//
// Image console interface.
//
// "add" outputs an image;
// "clear" clears the console;
// "update" is used to repaint the console before processing cycle end;
//
//================================================================

//================================================================
//
// ImgConsoleTarget
//
//================================================================

enum ImgOutputTarget {ImgOutputConsole, ImgOutputOverlay};

//================================================================
//
// ImgOutputHint
//
// Image description parameter group: name, id, minSize and newLine.
//
//================================================================

struct ImgOutputHint
{

public:

    FormatOutputAtom desc;
    uint32 id;
    Point<Space> minSize;
    ImgOutputTarget target;
    bool newLine;
    bool overlayCentering;
    float32 textFactor;
    float32 arrowFactor;

private:

    inline void initDefaultExceptDesc()
    {
        this->id = 0;
        this->minSize = point(0);
        this->newLine = false;
        this->target = ImgOutputOverlay;
        this->overlayCentering = false;
        this->textFactor = 1;
        this->arrowFactor = 1;
    }

public:

    inline ImgOutputHint(const FormatOutputAtom& desc)
        :
        desc(desc)
    {
        initDefaultExceptDesc();
    }

    inline ImgOutputHint(const CharArray& desc)
        :
        desc(desc)
    {
        initDefaultExceptDesc();
    }

    inline ImgOutputHint& setDesc(const FormatOutputAtom& desc)
        {this->desc = desc; return *this;}

    inline ImgOutputHint& setID(uint32 id)
        {this->id = id; return *this;}

    inline ImgOutputHint& setMinSize(const Point<Space>& minSize)
        {this->minSize = minSize; return *this;}

    inline ImgOutputHint& setNewLine(bool newLine = true)
        {this->newLine = newLine; return *this;}

    inline ImgOutputHint& setTarget(ImgOutputTarget target)
        {this->target = target; return *this;}

    inline ImgOutputHint& setTargetOverlay()
        {this->target = ImgOutputOverlay; return *this;}

    inline ImgOutputHint& setTargetConsole()
        {this->target = ImgOutputConsole; return *this;}

    inline ImgOutputHint& setOverlayCentering(bool overlayCentering = true)
        {this->overlayCentering = overlayCentering; return *this;}

    inline ImgOutputHint& setTextFactor(float32 factor)
        {this->textFactor = factor; return *this;}

    inline ImgOutputHint& setArrowFactor(float32 factor)
        {this->arrowFactor = factor; return *this;}

};
