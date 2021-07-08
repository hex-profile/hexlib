#include "baseImageConsoleToBridge.h"

#include "errorLog/blockExceptions.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsg.h"
#include "formatting/messageFormatter.h"

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
        : imageProvider(imageProvider), kit(kit), trace{trace} {}

    virtual void saveBgr32(ImageRef<PixelRgb32> dst)
    {
        auto code = [&] ()
        {
            auto matrix = Matrix<PixelRgb32>{dst.ptr, dst.pitch, dst.size.X, dst.size.Y, MatrixValidityAssertion{}};
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
            auto matrix = Matrix<PixelMono>{dst.ptr, dst.pitch, dst.size.X, dst.size.Y, MatrixValidityAssertion{}};
            require(imageProvider.saveBgr24(matrix, stdPass));
            returnTrue;
        };

        if_not (errorBlock(code()))
            throw CT("Debug bridge: Image provider: Saving image error");
    }

private:

    BaseImageProvider& imageProvider;
    Kit kit;
    TraceScope trace;

};

//================================================================
//
// BaseVideoOverlayToBridge::setImage
//
//================================================================

stdbool BaseVideoOverlayToBridge::setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
{
    BridgeImageProviderThunk bridgeProvider(imageProvider, stdPass);

    if_not (dataProcessing)
    {
        // Imitates the processing on counting phase. The pitch on execution phase may differ, 
        // but the provider implementation is tolerant to it up to some maximal row alignment.

        auto image = Matrix<uint8_x4>{nullptr, imageProvider.desiredPitch(), size.X, size.Y, MatrixValidityAssertion{}};
        require(imageProvider.saveBgr32(image, stdPass));
    }
    else
    {
        const Char* description = "";

        if (textEnabled)
        {
            kit.formatter.clear();
            formatOutput(desc, kit.formatter);
            REQUIRE(kit.formatter.valid());
            description = kit.formatter.data();
        }

        printMsg(kit.localLog, STR("OVERLAY: %"), desc);

        require(blockExceptionsVoid(destOverlay.set(ImagePoint{size.X, size.Y}, bridgeProvider, description)));
    }

    ////

    returnTrue;
}

//================================================================
//
// BaseVideoOverlayToBridge::setImageFake
//
//================================================================

stdbool BaseVideoOverlayToBridge::setImageFake(stdNullPars)
{
    returnTrue;
}

//================================================================
//
// BaseVideoOverlayToBridge::updateImage
//
//================================================================

stdbool BaseVideoOverlayToBridge::updateImage(stdNullPars) 
{
    REQUIRE(blockExceptionsVoid(destOverlay.update()));

    returnTrue;
}

//----------------------------------------------------------------

}
