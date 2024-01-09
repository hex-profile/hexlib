#include "guiModule.h"

#include "gui/drawTestImage/drawTestImage.h"
#include "gui/drawErrorPattern/drawErrorPattern.h"
#include "cfgTools/boolSwitch.h"
#include "errorLog/errorLog.h"
#include "gui/gpuConsoleDrawing/gpuConsoleDrawing.h"
#include "gui/gpuConsoleDrawing/fontMono9x16.h"
#include "numbers/divRound.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "gui/drawFilledRect/drawFilledRect.h"
#include "storage/rememberCleanup.h"
#include "numbers/mathIntrinsics.h"
#include "point/pointFunctions.h"

namespace gui {

//================================================================
//
// GuiModuleImpl
//
//================================================================

class GuiModuleImpl : public GuiModule
{

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit);

    ////

    virtual void extendMaxImageSize(const Point<Space>& size)
    {
        configMaxImageSize = maxv(configMaxImageSize, size);

        if_not (configMaxImageSize <= allocMaxImageSize)
            allocIsValid = false;
    }

    virtual OptionalObject<Point<Space>> getOverlayOffset() const
    {
        return mainImage.drawnOffset;
    }

    //----------------------------------------------------------------
    //
    // Realloc.
    //
    //----------------------------------------------------------------

    virtual bool reallocValid() const
        {return allocIsValid;}

    virtual stdbool realloc(stdPars(ReallocKit));

    //----------------------------------------------------------------
    //
    // Awake.
    //
    //----------------------------------------------------------------

    virtual stdbool checkWake(const CheckWakeArgs& args, stdPars(CheckWakeKit));

    virtual OptionalObject<TimeMoment> getWakeMoment() const {return wakeMoment;}

    //----------------------------------------------------------------
    //
    // Draw.
    //
    //----------------------------------------------------------------

    stdbool draw(const DrawArgs& args, stdPars(DrawKit));

    //----------------------------------------------------------------
    //
    // Redraw testing.
    //
    //----------------------------------------------------------------

    struct RedrawTest
    {
        BoolSwitch scrollOnRedraw{true};
        int32 scrollOffset = 0;
    }
    redrawTest;

    //----------------------------------------------------------------
    //
    // Alloc state.
    //
    //----------------------------------------------------------------

    bool allocIsValid = false;
    Point<Space> allocMaxImageSize = point(0);

    Point<Space> configMaxImageSize = point(0);

    //----------------------------------------------------------------
    //
    // Wake moment.
    //
    // It is used to indicate a moment of awakening.
    // If the value is present, it means that an animation is in progress
    // and the next change will occur at that moment.
    //
    //----------------------------------------------------------------

    OptionalObject<TimeMoment> wakeMoment;

    //----------------------------------------------------------------
    //
    // Mouse motion.
    //
    //----------------------------------------------------------------

    virtual void mouseButtonReceiver(const MouseButtonEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit))
    {
        if (event.button == 0 && event.press)
        {
            if (event.modifiers == 0)
            {
                mainImage.dragLastPos = event.position;
            }
            else if (event.modifiers == KeyModifier::Shift)
            {
                localConsole.dragLastPos = event.position;
                redraw.on = true;
            }
        }

        if (event.button == 0 && !event.press)
        {
            if (localConsole.dragLastPos)
                {localConsole.dragLastPos = {}; redraw.on = true;}

            mainImage.dragLastPos = {};
        }
    }

    virtual void mouseMoveReceiver(const MouseMoveEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit))
    {
        if (localConsole.dragLastPos)
        {
            auto& drag = *localConsole.dragLastPos;

            auto width = localConsoleGetWidth();
            width += (drag - event).X;

            if (lastOutputSize)
                localConsoleSetWidth(clampRange<float32>(width, 0, lastOutputSize->X));

            drag = event;
            redraw.on = true;
        }

        if (mainImage.dragLastPos && mainImage.divImageSize && mainImage.offsetMaxRadius)
        {
            auto& drag = *mainImage.dragLastPos;
            auto& R = *mainImage.offsetMaxRadius;
            auto& divImageSize = *mainImage.divImageSize;

            auto ofs = mainImage.normOffset();
            mainImage.normOffset = clampRange(ofs + divImageSize * (drag - event), -R, +R);
            drag = event;
            redraw.on = true;
        }
    }

    //----------------------------------------------------------------
    //
    // Font.
    //
    //----------------------------------------------------------------

    GpuFontMonoMemory gpuFontOwner;
    GpuFontMono gpuFont;

    //----------------------------------------------------------------
    //
    // Last displayed output size.
    //
    //----------------------------------------------------------------

    OptionalObject<Point<Space>> lastOutputSize;

    //----------------------------------------------------------------
    //
    // Global log.
    //
    //----------------------------------------------------------------

    struct GlobalConsole
    {
        NumericVar<float32> maxAge{0.f, typeMax<float32>(), 5.f};
        NumericVar<Point<Space>> border{point(0), point(typeMax<Space>()), point(4, 4)};
        GpuConsoleDrawer drawer;
    }
    globalConsole;

    //----------------------------------------------------------------
    //
    // Local log.
    //
    //----------------------------------------------------------------

    struct LocalConsole
    {
        NumericVar<float32> widthInChars{0, 8192, 60};
        NumericVar<Point<Space>> border{point(0), point(typeMax<Space>()), point(4, 4)};
        GpuConsoleDrawer drawer;
        OptionalObject<Point<float32>> dragLastPos;
    }
    localConsole;

    ////

    float32 localConsoleGetWidth() const
        {return localConsole.widthInChars * gpuFont.charSize.X;}

    void localConsoleSetWidth(float32 width)
        {localConsole.widthInChars = width / gpuFont.charSize.X;}

    //----------------------------------------------------------------
    //
    // Main image.
    //
    //----------------------------------------------------------------

    struct MainImage
    {
        OptionalObject<Point<float32>> divImageSize;
        OptionalObject<Point<float32>> offsetMaxRadius;

        NumericVar<Point<float32>> normOffset{point(-0.5f), point(+0.5f), point(0.f)};
        OptionalObject<Point<Space>> drawnOffset;

        OptionalObject<Point<float32>> dragLastPos;
    };

    MainImage mainImage;

    //----------------------------------------------------------------
    //
    // Colors
    //
    //----------------------------------------------------------------

    struct Lines
    {
        NumericVar<Point3D<float32>> bodyColorCfg{point3D(0.f), point3D(1.f), point3D(1.f)};
        NumericVar<Point3D<float32>> shadowColorCfg{point3D(0.f), point3D(1.f), point3D(0.f)};

        auto bodyColor() const {return convertNearest<uint8_x4>(0xFF * bodyColorCfg());}
        auto shadowColor() const {return convertNearest<uint8_x4>(0xFF * shadowColorCfg());}

        NumericVar<Space> shadowRadius{0, 1024, 1};
        NumericVar<Space> lineWidth{0, 1024, 2};
        NumericVar<Space> scrollbarWidth{0, 1024, 2};
    };

    Lines lines;

};

////

UniquePtr<GuiModule> GuiModule::create()
{
    return makeUnique<GuiModuleImpl>();
}

//================================================================
//
// GuiModuleImpl::serialize
//
//================================================================

void GuiModuleImpl::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("GUI Debug");

        {
            CFG_NAMESPACE("Redraw Testing");

            redrawTest.scrollOnRedraw.serialize(kit, STR("Scroll On Redraw"));
        }
    }

    {
        CFG_NAMESPACE("Global Log");
        globalConsole.maxAge.serialize(kit, STR("Message Display Time In Seconds"));
        globalConsole.border.serialize(kit, STR("Border In Pixels"));
        globalConsole.drawer.serialize(kit);
    }

    {
        CFG_NAMESPACE("Local Log");

        localConsole.widthInChars.serialize(kit, STR("Width In Chars"));
        localConsole.border.serialize(kit, STR("Border In Pixels"));
        localConsole.drawer.serialize(kit);
    }

    {
        CFG_NAMESPACE("Main Image");

        mainImage.normOffset.serialize(kit, STR("Normalized Offset"),
            STR("The image is drawn shifted by the specified fraction of its size"));
    }

    {
        CFG_NAMESPACE("Lines");
        lines.bodyColorCfg.serialize(kit, STR("Body Color"));
        lines.shadowColorCfg.serialize(kit, STR("Shadow Color"));
        lines.shadowRadius.serialize(kit, STR("Shadow Radius In Pixels"));
        lines.lineWidth.serialize(kit, STR("Line Width In Pixels"));
        lines.scrollbarWidth.serialize(kit, STR("Scrollbar Width In Pixels"));
    }
}

//================================================================
//
// GuiModuleImpl::realloc
//
//================================================================

stdbool GuiModuleImpl::realloc(stdPars(ReallocKit))
{
    allocIsValid = false;

    ////

    auto font = fontMono9x16();

    require(gpuFontOwner.realloc(font, stdPass));
    gpuFont = gpuFontOwner;

    ////

    REQUIRE(font.charSize >= 1);
    REQUIRE(configMaxImageSize >= 0);

    auto textBufferSize = divUpNonneg(configMaxImageSize, font.charSize);
    textBufferSize = clampMin(textBufferSize, 1);

    ////

    require(globalConsole.drawer.realloc(textBufferSize, stdPass));

    ////

    require(localConsole.drawer.realloc(textBufferSize, stdPass));

    ////

    allocIsValid = true;
    allocMaxImageSize = configMaxImageSize;

    returnTrue;
}

//================================================================
//
// GuiModuleImpl::checkWake
//
//================================================================

stdbool GuiModuleImpl::checkWake(const CheckWakeArgs& args, stdPars(CheckWakeKit))
{
    //----------------------------------------------------------------
    //
    // Check log modification moment.
    //
    //----------------------------------------------------------------

    auto currentMoment = kit.timer.moment();

    auto logModification = args.globalLog.getLastModification();

    if (logModification && kit.timer.diff(*logModification, currentMoment) <= globalConsole.maxAge())
    {
        if_not (wakeMoment) // Set to wake immediately, on redraw it will set the moment precisely.
            wakeMoment = currentMoment;
    }

    ////

    returnTrue;
}

//================================================================
//
// GuiModuleImpl::draw
//
//================================================================

stdbool GuiModuleImpl::draw(const DrawArgs& args, stdPars(DrawKit))
{
    REQUIRE(allocIsValid);

    ////

    auto outputImage = args.dstImage;
    auto outputSize = args.dstImage.size();

    lastOutputSize = outputSize;

    //----------------------------------------------------------------
    //
    // Test image.
    //
    //----------------------------------------------------------------

    if (redrawTest.scrollOnRedraw)
    {
        if (kit.dataProcessing)
            redrawTest.scrollOffset += 1;
    }

    require(drawTestImage(point(redrawTest.scrollOffset, 0), 8, 16, outputImage, stdPass));

    //----------------------------------------------------------------
    //
    // Draw main image.
    //
    //----------------------------------------------------------------

    if (args.overlay.hasUpdates())
    {
        auto imageUser = overlayBuffer::ImageUser::O | [&] (bool valid, auto& image, stdParsNull)
        {
            auto usedSize = minv(image.size(), outputSize);

            GpuMatrix<uint8_x4> dstRegion;
            require(outputImage.subs(point(0), usedSize, dstRegion));

            ////

            auto shiftSize = image.size() - usedSize;
            REQUIRE(shiftSize >= 0); // valid shift range [0, shiftSize]

            auto shiftSizef = convertFloat32(shiftSize);
            auto divShiftSize = fastRecipZero(shiftSizef);
            auto shiftHalf = 0.5f * shiftSizef;


            ////

            auto imageSizef = convertFloat32(image.size());
            auto shiftf = shiftHalf + mainImage.normOffset() * imageSizef;
            shiftf = clampRange(shiftf, 0.f, shiftSizef);
            auto shifti = clampRange(convertNearest<Space>(shiftf), 0, shiftSize);

            ////

            auto divImageSize = fastRecipZero(imageSizef);
            mainImage.divImageSize = divImageSize;
            mainImage.offsetMaxRadius = divImageSize * shiftHalf;
            mainImage.drawnOffset = shifti;

            ////

            GpuMatrix<const uint8_x4> srcRegion;
            REQUIRE(image.subs(shifti, usedSize, srcRegion));

            require(gpuMatrixCopy(srcRegion, dstRegion, stdPass));

            ////

            auto visibleFraction = usedSize * divImageSize;
            auto scroll = image.size() > usedSize;

            ////

            auto barSize = clampRange(convertNearest<Space>(visibleFraction * outputSize), 0, outputSize);
            auto barSpace = outputSize - barSize; // >= 0

            ////

            auto ofsAlpha = saturatev(convertFloat32(shifti) * divShiftSize);
            auto barOfs = clampRange(convertNearest<Space>(barSpace * ofsAlpha), 0, barSpace);
            auto barThickess = lines.scrollbarWidth;

            ////

            auto horOrg = point(barOfs.X, outputSize.Y - barThickess);
            auto horEnd = point(barOfs.X + barSize.X, outputSize.Y);

            auto verOrg = point(outputSize.X - barThickess, barOfs.Y);
            auto verEnd = point(outputSize.X, barOfs.Y + barSize.Y);

            auto bodyColor = lines.bodyColor();
            auto shadowColor = lines.shadowColor();

            if (scroll.X)
                require(drawSingleRect({horOrg - lines.shadowRadius(), horEnd + lines.shadowRadius(), shadowColor}, outputImage, stdPass));

            if (scroll.Y)
                require(drawSingleRect({verOrg - lines.shadowRadius(), verEnd + lines.shadowRadius(), shadowColor}, outputImage, stdPass));

            if (scroll.X)
                require(drawSingleRect({horOrg, horEnd, bodyColor}, outputImage, stdPass));

            if (scroll.Y)
                require(drawSingleRect({verOrg, verEnd, bodyColor}, outputImage, stdPass));

            ////

            returnTrue;
        };

        errorBlock(args.overlay.useImage(imageUser, stdPassNc));
    }

    //----------------------------------------------------------------
    //
    // Text tools.
    //
    //----------------------------------------------------------------

    auto addTextRow = [] (auto& text, auto& kind, auto& moment, auto& destination)
    {
        uint32 color = 0x00FFFFFF;
        if (kind == msgWarn) color = 0x00FFFF00;
        if (kind == msgErr) color = 0x00FF8040;

        destination(text, color);
    };

    //----------------------------------------------------------------
    //
    // Global console.
    //
    //----------------------------------------------------------------

    auto localConsoleSpace = clampRange(convertNearest<Space>(localConsoleGetWidth()), 0, outputSize.X);

    ////

    auto clampByOutput = [&] (const auto& value)
        {return clampRange(value, point(0), outputSize);};

    ////

    auto globalConsoleOrg = point(0);
    auto globalConsoleEnd = outputSize;
    globalConsoleEnd.X -= localConsoleSpace;

    globalConsoleOrg = clampByOutput(globalConsoleOrg + globalConsole.border());
    globalConsoleEnd = clampByOutput(globalConsoleEnd - globalConsole.border());

    ////

    OptionalObject<TimeMoment> oldestMessage;

    ////

    if (allv(globalConsoleOrg < globalConsoleEnd))
    {
        auto currentMoment = kit.timer.moment();

        auto textProvider = gpuConsoleDrawing::ColorTextProvider::O | [&] (auto maxCount, auto& colorTextReceiver, stdParsNull)
        {
            REQUIRE(kit.dataProcessing);
            REQUIRE(maxCount >= 0);

            auto logBufferReceiver = LogBufferReceiver::O | [&] (auto& text, auto& kind, auto& moment)
            {
                if_not (kit.timer.diff(moment, currentMoment) <= globalConsole.maxAge())
                    return;

                if_not (oldestMessage)
                    oldestMessage = moment;

                addTextRow(text, kind, moment, colorTextReceiver);
            };

            args.globalLog.readLastMessages(logBufferReceiver, maxCount);
            returnTrue;
        };

        require(globalConsole.drawer.drawText(textProvider, outputImage, globalConsoleOrg, globalConsoleEnd, gpuFont, nullptr, stdPass));

        //
        // Animation.
        //

        if (kit.dataProcessing)
        {
            if_not (oldestMessage)
                wakeMoment = {};
            else
                wakeMoment = kit.timer.add(*oldestMessage, globalConsole.maxAge());
        }
    }

    //----------------------------------------------------------------
    //
    // Local console.
    //
    //----------------------------------------------------------------

    auto localConsoleSpaceOrg = clampByOutput(point(outputSize.X - localConsoleSpace, 0));
    auto localConsoleSpaceEnd = outputSize;

    auto localConsoleOrg = clampByOutput(localConsoleSpaceOrg + localConsole.border());
    auto localConsoleEnd = clampByOutput(localConsoleSpaceEnd - localConsole.border());

    if (allv(localConsoleOrg < localConsoleEnd))
    {
        auto textProvider = gpuConsoleDrawing::ColorTextProvider::O | [&] (auto maxCount, auto& colorTextReceiver, stdParsNull)
        {
            REQUIRE(kit.dataProcessing);
            REQUIRE(maxCount >= 0);

            auto logBufferReceiver = LogBufferReceiver::O | [&] (auto& text, auto& kind, auto& moment)
            {
                addTextRow(text, kind, moment, colorTextReceiver);
            };

            args.localLog.readFirstMessagesShowOverflow(logBufferReceiver, maxCount);
            returnTrue;
        };

        require(localConsole.drawer.drawText(textProvider, outputImage, localConsoleOrg, localConsoleEnd, gpuFont, nullptr, stdPass));
    }

    ////

    if (localConsole.dragLastPos)
    {
        auto org = localConsoleSpaceOrg;
        auto end = point(org.X, outputSize.Y);

        auto lineWidth = lines.lineWidth();
        auto lineWidthHalf = lineWidth / 2;

        org.X -= lineWidthHalf;
        end.X += (lineWidth - lineWidthHalf);

        auto shadowRadius = lines.shadowRadius();

        DrawFilledRectArgs args
        {
            {org, end, lines.bodyColor()},
            {org - shadowRadius, end + shadowRadius, lines.shadowColor()}
        };

        errorBlock(drawFilledRect(args, outputImage, stdPassNc));
    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
