#include "guiModule.h"

#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "errorLog/errorLog.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "gui/drawBackgroundPattern/drawBackgroundPattern.h"
#include "gui/drawErrorPattern/drawErrorPattern.h"
#include "gui/drawFilledRect/drawFilledRect.h"
#include "gui/gpuConsoleDrawing/fontMono9x16.h"
#include "gui/gpuConsoleDrawing/gpuConsoleDrawing.h"
#include "numbers/divRound.h"
#include "numbers/mathIntrinsics.h"
#include "point/pointFunctions.h"
#include "storage/rememberCleanup.h"

namespace gui {

using gpuConsoleDrawing::PadMode;
using gpuConsoleDrawing::DrawText;

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
    {
        return allocIsValid;
    }

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
    // Desktop settings.
    //
    //----------------------------------------------------------------

    enum class DesktopMode {RandomPattern, SolidColor, COUNT};

    struct Desktop
    {
        MultiSwitch<DesktopMode, DesktopMode::COUNT, DesktopMode::RandomPattern> mode;
        NumericVar<Point3D<float32>> solidColor{point3D(0.f), point3D(1.f), point3D(1.f)};

        BoolSwitch scrollOnRedraw{false};
        int32 scrollOffset = 0;
    }
    desktop;

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

    virtual stdbool mouseButtonReceiver(const MouseButtonEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit))
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

        returnTrue;
    }

    virtual stdbool mouseMoveReceiver(const MouseMoveEvent& event, RedrawRequest& redraw, stdPars(ErrorLogKit))
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

        returnTrue;
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
    // Text logs config.
    //
    //----------------------------------------------------------------

    struct TextLogs
    {
        NumericVar<Space> fontUpscalingFactor{1, 16, 1};

        NumericVar<Point<Space>> border{point(0), point(typeMax<Space>()), point(6, 3)};

        NumericVar<Point3D<float32>> textColorInfo{point3D(0.f), point3D(1.f), point3D(1.f)};
        NumericVar<Point3D<float32>> textColorWarn{point3D(0.f), point3D(1.f), point3D(0.f, 1.f, 1.f)};
        NumericVar<Point3D<float32>> textColorErr{point3D(0.f), point3D(1.f), point3D(0.25f, 0.25f, 1.f)};

        NumericVar<Point3D<float32>> outlineColor{point3D(0.f), point3D(1.f), point3D(0.f)};
    }
    textLogs;

    //----------------------------------------------------------------
    //
    // Global log.
    //
    //----------------------------------------------------------------

    struct GlobalConsole
    {
        NumericVar<float32> maxAge{0.f, typeMax<float32>(), 10.f};
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
        NumericVar<float32> widthInPixels{0, 8192, 540};

        MultiSwitch<PadMode, PadMode::COUNT, PadMode::None> padMode;
        NumericVar<Point3D<float32>> padColor{point3D(0.f), point3D(1.f), point3D(0.f)};
        NumericVar<float32> padOpacity{0, 1, 0.5f};

        GpuConsoleDrawer drawer;
        OptionalObject<Point<float32>> dragLastPos;
    }
    localConsole;

    ////

    float32 localConsoleGetWidth() const
        {return localConsole.widthInPixels;}

    void localConsoleSetWidth(float32 width)
        {localConsole.widthInPixels = width;}

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

    struct Common
    {
        NumericVar<Point3D<float32>> bodyColor{point3D(0.f), point3D(1.f), point3D(1.f)};
        NumericVar<Point3D<float32>> shadowColor{point3D(0.f), point3D(1.f), point3D(0.f)};

        NumericVar<Space> lineWidth{0, 1024, 2};
        NumericVar<Space> scrollbarWidth{0, 1024, 2};

        NumericVar<Space> shadowRadius{0, 1024, 1};
    };

    Common common;

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
        CFG_NAMESPACE("Desktop");

        desktop.mode.serialize(kit, STR("Mode"), STR("Random Pattern"), STR("Solid Color"));
        desktop.solidColor.serialize(kit, STR("Solid Color"));
        desktop.scrollOnRedraw.serialize(kit, STR("Scroll On Redraw (For Debugging)"));
    }

    {
        CFG_NAMESPACE("Scrollbars And Dragging Lines");

        common.bodyColor.serialize(kit, STR("Body Color"));
        common.shadowColor.serialize(kit, STR("Shadow Color"));

        common.lineWidth.serialize(kit, STR("Line Width"));
        common.scrollbarWidth.serialize(kit, STR("Scrollbar Width"));

        common.shadowRadius.serialize(kit, STR("Shadow Radius"));
    }

    {
        CFG_NAMESPACE("Main Image");

        mainImage.normOffset.serialize(kit, STR("Normalized Offset"), STR(""),
            STR("The image is drawn shifted by the specified fraction of its size"));
    }

    {
        CFG_NAMESPACE("Text Logs");

        bool fontSizeSteady = textLogs.fontUpscalingFactor.serialize(kit, STR("Font Upscaling Factor"),
            STR(""), STR("Use value `2` as a temporary solution for 4k monitors"));

        if_not (fontSizeSteady)
            allocIsValid = false;

        textLogs.border.serialize(kit, STR("Border In Pixels"));

        textLogs.textColorInfo.serialize(kit, STR("Text Color: Info"));
        textLogs.textColorWarn.serialize(kit, STR("Text Color: Warning"));
        textLogs.textColorErr.serialize(kit, STR("Text Color: Error"));

        textLogs.outlineColor.serialize(kit, STR("Outline Color"));

        {
            CFG_NAMESPACE("Global Log");
            globalConsole.maxAge.serialize(kit, STR("Message Display Time In Seconds"));
            globalConsole.drawer.serialize(kit);
        }

        {
            CFG_NAMESPACE("Local Log");

            localConsole.widthInPixels.serialize(kit, STR("Width In Pixels"));
            localConsole.padMode.serialize(kit, STR("Pad Mode"), STR("None"), STR("Used Space"), STR("Entire Space"));
            localConsole.padColor.serialize(kit, STR("Pad Color"));
            localConsole.padOpacity.serialize(kit, STR("Pad Opacity"));
            localConsole.drawer.serialize(kit);
        }
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

    auto textBufferSize = divUpNonneg(configMaxImageSize, font.charSize * textLogs.fontUpscalingFactor);
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
    // convertColor
    //
    //----------------------------------------------------------------

    auto convertColor = [] (const Point3D<float32>& color)
    {
        auto vec = convertNearest<float32_x4>(color);
        return convertNormClamp<uint8_x4>(vec);
    };

    ////

    auto bodyColor = convertColor(common.bodyColor);
    auto shadowColor = convertColor(common.shadowColor);

    //----------------------------------------------------------------
    //
    // Desktop.
    //
    //----------------------------------------------------------------

    if (desktop.scrollOnRedraw)
    {
        if (kit.dataProcessing)
            desktop.scrollOffset += 1;
    }

    auto drawDesktop = [&] (stdPars(auto))
    {
        if (desktop.mode == DesktopMode::RandomPattern)
        {
            require(drawBackgroundPattern(point(desktop.scrollOffset, 0), outputImage, stdPass));
        }
        else if (desktop.mode == DesktopMode::SolidColor)
        {
            auto color = convertColor(desktop.solidColor);
            require(gpuMatrixSet(outputImage, color, stdPass));
        }
        else
        {
            REQUIRE(false);
        }

        returnTrue;
    };

    errorBlock(drawDesktop(stdPassNc));

    //----------------------------------------------------------------
    //
    // Main image scrollbars: Draw at the end.
    //
    //----------------------------------------------------------------

    auto sbarFlags = point(false);

    auto sbarHorOrg = point(0);
    auto sbarHorEnd = point(0);

    auto sbarVerOrg = point(0);
    auto sbarVerEnd = point(0);

    ////

    auto drawScrollbars = [&] (stdPars(auto))
    {
        if (sbarFlags.X)
            require(drawSingleRect({sbarHorOrg - common.shadowRadius(), sbarHorEnd + common.shadowRadius(), shadowColor}, outputImage, stdPass));

        if (sbarFlags.Y)
            require(drawSingleRect({sbarVerOrg - common.shadowRadius(), sbarVerEnd + common.shadowRadius(), shadowColor}, outputImage, stdPass));

        if (sbarFlags.X)
            require(drawSingleRect({sbarHorOrg, sbarHorEnd, bodyColor}, outputImage, stdPass));

        if (sbarFlags.Y)
            require(drawSingleRect({sbarVerOrg, sbarVerEnd, bodyColor}, outputImage, stdPass));

        returnTrue;
    };

    REMEMBER_CLEANUP(errorBlock(drawScrollbars(stdPassNc)));

    //----------------------------------------------------------------
    //
    // Main image.
    //
    //----------------------------------------------------------------

    auto drawMainImage = [&] (stdPars(auto))
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
            sbarFlags = image.size() > usedSize;

            ////

            auto barSize = clampRange(convertNearest<Space>(visibleFraction * outputSize), 0, outputSize);
            auto barSpace = outputSize - barSize; // >= 0

            ////

            auto ofsAlpha = saturatev(convertFloat32(shifti) * divShiftSize);
            auto barOfs = clampRange(convertNearest<Space>(barSpace * ofsAlpha), 0, barSpace);
            auto barThickess = common.scrollbarWidth;

            ////

            sbarHorOrg = point(barOfs.X, outputSize.Y - barThickess);
            sbarHorEnd = point(barOfs.X + barSize.X, outputSize.Y);

            sbarVerOrg = point(outputSize.X - barThickess, barOfs.Y);
            sbarVerEnd = point(outputSize.X, barOfs.Y + barSize.Y);

            ////

            returnTrue;
        };

        if (args.overlay.hasUpdates())
            require(args.overlay.useImage(imageUser, stdPass));

        returnTrue;
    };

    ////

    errorBlock(drawMainImage(stdPassNc));

    //----------------------------------------------------------------
    //
    // Text tools.
    //
    //----------------------------------------------------------------

    auto getConsoleColor = [&] (const Point3D<float32>& color) -> uint32
    {
        auto color8u = convertColor(color);
        return recastEqualLayout<uint32>(color8u);
    };

    ////

    auto conColorInfo = getConsoleColor(textLogs.textColorInfo);
    auto conColorWarn = getConsoleColor(textLogs.textColorWarn);
    auto conColorErr = getConsoleColor(textLogs.textColorErr);

    ////

    auto addTextRow = [&] (auto& text, auto& kind, auto& moment, auto& destination)
    {
        uint32 color = conColorInfo;
        if (kind == msgWarn) color = conColorWarn;
        if (kind == msgErr) color = conColorErr;

        destination(text, color);
    };

    //----------------------------------------------------------------
    //
    // Global console.
    //
    //----------------------------------------------------------------

    auto clampByOutput = [&] (const auto& value)
        {return clampRange(value, point(0), outputSize);};

    ////

    auto localConsoleSpace = clampRange(convertNearest<Space>(localConsoleGetWidth()), 0, outputSize.X);

    ////

    auto drawGlobalConsole = [&] (stdPars(auto))
    {
        auto globalConsoleOrg = point(0);
        auto globalConsoleEnd = clampByOutput(outputSize - point(localConsoleSpace, 0));

        ////

        OptionalObject<TimeMoment> oldestMessage;

        ////

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

        ////

        DrawText args;
        args.buffer = &textProvider;
        args.destination = outputImage;
        args.renderOrg = globalConsoleOrg;
        args.renderEnd = globalConsoleEnd;
        args.border = textLogs.border;
        args.font = &gpuFont;
        args.fontUpscalingFactor = textLogs.fontUpscalingFactor;
        args.outlineColor = textLogs.outlineColor;

        require(globalConsole.drawer.drawText(args, stdPass));

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

        returnTrue;
    };

    ////

    errorBlock(drawGlobalConsole(stdPassNc));

    //----------------------------------------------------------------
    //
    // Local console.
    //
    //----------------------------------------------------------------

    auto localConsoleOrg = clampByOutput(point(outputSize.X - localConsoleSpace, 0));
    auto localConsoleEnd = outputSize;

    ////

    auto drawLocalConsole = [&] (stdPars(auto))
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

        //----------------------------------------------------------------
        //
        // Draw text.
        //
        //----------------------------------------------------------------

        DrawText args;
        args.buffer = &textProvider;
        args.destination = outputImage;
        args.renderOrg = localConsoleOrg;
        args.renderEnd = localConsoleEnd;
        args.border = textLogs.border;
        args.font = &gpuFont;
        args.fontUpscalingFactor = textLogs.fontUpscalingFactor;
        args.padMode = localConsole.padMode;
        args.padColor = localConsole.padColor;
        args.padOpacity = localConsole.padOpacity;
        args.outlineColor = textLogs.outlineColor;

        require(localConsole.drawer.drawText(args, stdPass));

        returnTrue;
    };

    ////

    errorBlock(drawLocalConsole(stdPassNc));

    //----------------------------------------------------------------
    //
    // Local console dragging.
    //
    //----------------------------------------------------------------

    auto drawLocalConsoleDragging = [&] (stdPars(auto))
    {
        if_not (localConsole.dragLastPos)
            returnTrue;

        ////

        auto org = localConsoleOrg;
        auto end = point(org.X, outputSize.Y);

        auto lineWidth = common.lineWidth();
        auto lineWidthHalf = lineWidth / 2;

        org.X -= lineWidthHalf;
        end.X += (lineWidth - lineWidthHalf);

        auto shadowRadius = common.shadowRadius();

        DrawFilledRectArgs args
        {
            {org, end, bodyColor},
            {org - shadowRadius, end + shadowRadius, shadowColor}
        };

        require(drawFilledRect(args, outputImage, stdPass));

        returnTrue;
    };

    ////

    errorBlock(drawLocalConsoleDragging(stdPassNc));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}

