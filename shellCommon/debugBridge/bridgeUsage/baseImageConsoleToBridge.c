#include "baseImageConsoleToBridge.h"

#include "charType/strUtils.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/errorLog.h"
#include "formatting/messageFormatter.h"
#include "userOutput/printMsg.h"

namespace baseImageConsoleToBridge {

using debugBridge::ImagePoint;
using debugBridge::ImageRef;
using debugBridge::ImageSpace;
using debugBridge::PixelRgb32;
using debugBridge::PixelMono;
using debugBridge::Char;

//================================================================
//
// BridgeImageProviderThunk
//
//================================================================

class BridgeImageProviderThunk : public debugBridge::ImageProvider
{

public:

    BridgeImageProviderThunk(BaseImageProvider& imageProvider, stdPars(Kit))
        : imageProvider(imageProvider), stdParsCapture {}

    virtual void saveBgr32(ImageRef<PixelRgb32> dst)
    {
        auto code = [&] ()
        {
            Matrix<PixelRgb32> matrix;
            REQUIRE(matrix.assignValidated(dst.ptr, dst.pitch, dst.size.X, dst.size.Y));
            require(imageProvider.saveBgr32(recastElement<uint8_x4>(matrix), stdPass));
            returnTrue;
        };

        if_not (errorBlock(code()))
            throw CT("Debug bridge: Image provider: Saving image error");
    }

    virtual void saveBgr24(ImageRef<PixelMono> dst)
    {
        auto code = [&] ()
        {
            Matrix<PixelMono> matrix;
            REQUIRE(matrix.assignValidated(dst.ptr, dst.pitch, dst.size.X, dst.size.Y));
            require(imageProvider.saveBgr24(matrix, stdPass));
            returnTrue;
        };

        if_not (errorBlock(code()))
            throw CT("Debug bridge: Image provider: Saving image error");
    }

private:

    BaseImageProvider& imageProvider;
    stdParsMember(Kit);

};

//================================================================
//
// BaseVideoOverlayToBridge::overlayClear
//
//================================================================

stdbool BaseVideoOverlayToBridge::overlayClear(stdParsNull)
{
    convertExceptions(destOverlay.clear());
    returnTrue;
}

//================================================================
//
// BaseVideoOverlayToBridge::overlaySet
//
//================================================================

stdbool BaseVideoOverlayToBridge::overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
{
    BridgeImageProviderThunk bridgeProvider(imageProvider, stdPass);

    if_not (dataProcessing)
    {
        // Imitates the processing on counting phase. The pitch on execution phase may differ,
        // but the provider implementation is tolerant to it up to some maximal row alignment.

        Matrix<uint8_x4> image;
        REQUIRE(image.assignValidated(nullptr, imageProvider.desiredPitch(), size.X, size.Y));
        require(imageProvider.saveBgr32(image, stdPass));
    }
    else
    {
        if (textEnabled)
            printMsg(kit.localLog, STR("OVERLAY: %"), desc);

        convertExceptions(destOverlay.set(ImagePoint{size.X, size.Y}, bridgeProvider, ""));
    }

    ////

    returnTrue;
}

//================================================================
//
// BaseVideoOverlayToBridge::overlaySetFake
//
//================================================================

stdbool BaseVideoOverlayToBridge::overlaySetFake(stdParsNull)
{
    returnTrue;
}

//================================================================
//
// BaseVideoOverlayToBridge::overlayUpdate
//
//================================================================

stdbool BaseVideoOverlayToBridge::overlayUpdate(stdParsNull)
{
    returnTrue;
}

//----------------------------------------------------------------

}
