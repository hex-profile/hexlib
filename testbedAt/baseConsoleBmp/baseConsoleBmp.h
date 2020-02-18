#pragma once

#include "allocation/mallocKit.h"
#include "baseImageConsole/baseImageConsole.h"
#include "interfaces/fileTools.h"
#include "kits/msgLogsKit.h"
#include "storage/dynamicClass.h"
#include "userOutput/diagnosticKit.h"

namespace baseConsoleBmp {

//================================================================
//
// Pixel
//
//================================================================

using Pixel = uint8_x4;

//================================================================
//
// Kit
//
//================================================================

using Kit = KitCombine<DiagnosticKit, FileToolsKit, MallocKit>;

//================================================================
//
// BaseConsoleBmp
//
//================================================================

class BaseConsoleBmp
{

public:

    BaseConsoleBmp();
    ~BaseConsoleBmp();

    stdbool saveImage(const Matrix<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool setOutputDir(const CharType* outputDir, stdPars(Kit));

private:
                
    DynamicClass<class BaseConsoleBmpImpl> instance;

};

//================================================================
//
// BaseConsoleBmpThunk
//
// Temporal thunk: adds necessary kit to BaseConsoleBmp
//
//================================================================

class BaseConsoleBmpThunk : public BaseImageConsole, public BaseVideoOverlay
{

public:

    BaseConsoleBmpThunk(BaseConsoleBmp& saver, BaseImageConsole& baseConsole, BaseVideoOverlay& baseOverlay, const Kit& kit)
        : saver(saver), baseConsole(baseConsole), baseOverlay(baseOverlay), kit(kit) {}

public:

    stdbool addImage(const Matrix<const Pixel>& img, const ImgOutputHint& hint, stdNullPars)
    {
        bool ok1 = errorBlock(baseConsole.addImage(img, hint, stdPass));
        bool ok2 = errorBlock(saver.saveImage(img, hint.desc, hint.id, stdPass));
        require(ok1 && ok2);
        returnTrue;
    }

    stdbool clear(stdNullPars)
    {
        return baseConsole.clear(stdPassThru);
    }

    stdbool update(stdNullPars)
    {
        return baseConsole.update(stdPassThru);
    }

public:

    stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        bool ok1 = errorBlock(baseOverlay.setImage(size, imageProvider, desc, id, textEnabled, stdPass));
        bool ok2 = errorBlock(saver.saveImage(size, imageProvider, desc, id, stdPass));
        require(ok1 && ok2);
        returnTrue;
    }

    stdbool setImageFake(stdNullPars)
    {
        return baseOverlay.setImageFake(stdPassThru);
    }

    stdbool updateImage(stdNullPars) 
    {
        returnTrue;
    }

private:

    BaseConsoleBmp& saver;
    BaseImageConsole& baseConsole;
    BaseVideoOverlay& baseOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using baseConsoleBmp::BaseConsoleBmp;
using baseConsoleBmp::BaseConsoleBmpThunk;
