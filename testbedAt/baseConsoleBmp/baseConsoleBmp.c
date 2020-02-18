#include "baseConsoleBmp.h"

#include <string>
#include <map>
#include <sstream>
#include <iomanip>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "errorLog/errorLog.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "flipMatrix.h"
#include "formattedOutput/formatStreamStl.h"
#include "data/spacex.h"
#include "baseImageConsole/imageProviderMemcpy.h"

namespace baseConsoleBmp {

using namespace std;

//================================================================
//
// convertControlled
//
//================================================================

template <typename Src, typename Dst>
bool convertControlled(Src src, Dst& dst);

//----------------------------------------------------------------

bool convertControlled(Space src, DWORD& dst)
{
    ensure(src >= 0 && src <= 0x7FFFFFFF); // DWORD is signed 32-bit
    dst = src;
    return true;
}

//----------------------------------------------------------------

bool convertControlled(Space src, LONG& dst)
{
    ensure(src >= LONG_MIN && src <= LONG_MAX); // LONG is signed 32-bit
    dst = src;
    return true;
}

//================================================================
//
// ASSIGN_CONVERT
//
//================================================================

#define ASSIGN_CONVERT(dst, src) \
    REQUIRE(convertControlled(src, dst));

//================================================================
//
// String
//
//================================================================

using String = basic_string<CharType>;

//================================================================
//
// mapToFilename
//
//================================================================

String mapToFilename(const String& s)
{
    String result = s;

    ////

    for (String::iterator i = result.begin(); i != result.end(); ++i)
    {
        CharType c = *i;

        if_not
        (
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            (c == '_')
        )
            c = '-';

        *i = c;
    }

    ////

    return result;
}

//================================================================
//
// formatAtomToString
//
//================================================================

stdbool formatAtomToString(const FormatOutputAtom& v, String& result, stdPars(ErrorLogKit))
{
    std::basic_stringstream<CharType> stringStream;
    FormatStreamStlThunk formatToStream(stringStream);

    v.func(v.value, formatToStream);
    REQUIRE(formatToStream.isOk());
    REQUIRE(!!stringStream);

    result = stringStream.str();

    returnTrue;
}

//================================================================
//
// bmpAlignment
//
//================================================================

static const size_t bmpAlignmentMask = 3;

//================================================================
//
// getAlignedPitch
//
//================================================================

stdbool getAlignedPitch(Space sizeX, Space& pitch, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0);

    REQUIRE(sizeX <= spaceMax / Space(sizeof(Pixel)));
    Space rowMemSize = sizeX * Space(sizeof(Pixel));

    REQUIRE(rowMemSize <= spaceMax - bmpAlignmentMask);
    Space rowAlignedSize = (rowMemSize + bmpAlignmentMask) & (~bmpAlignmentMask);

    Space bufSizeX = rowAlignedSize / Space(sizeof(Pixel));
    REQUIRE(bufSizeX * Space(sizeof(Pixel)) == rowAlignedSize);

    pitch = bufSizeX;

    returnTrue;
}

//================================================================
//
// BitmapheaderPalette
//
//================================================================

struct BitmapinfoPalette : public BITMAPINFO
{
    RGBQUAD additionalColors[255];
};

//================================================================
//
// makeBitmapHeader
//
//================================================================

template <typename Pixel>
stdbool makeBitmapHeader(const Point<Space>& size, BitmapinfoPalette& result, stdPars(ErrorLogKit))
{
    BITMAPINFOHEADER& bmi = result.bmiHeader;

    ////

    REQUIRE(size.X >= 0 && size.Y >= 0);

    //
    // check dimensions
    //

    Space alignedPitch = 0;
    require(getAlignedPitch(size.X, alignedPitch, stdPass));

    //
    // fill the structure
    //

    bmi.biSize = sizeof(BITMAPINFOHEADER);
    ASSIGN_CONVERT(bmi.biWidth, size.X);
    ASSIGN_CONVERT(bmi.biHeight, size.Y);
    bmi.biPlanes = 1;
    bmi.biBitCount = sizeof(Pixel) * 8;
    bmi.biCompression = BI_RGB;
    bmi.biSizeImage = 0;
    bmi.biXPelsPerMeter = 3200;
    bmi.biYPelsPerMeter = 3200;
    bmi.biClrUsed = 0;
    bmi.biClrImportant = 0;

    //
    //
    //

    for (int32 i = 0; i < 256; ++i)
    {
        result.bmiColors[i].rgbRed = i;
        result.bmiColors[i].rgbGreen = i;
        result.bmiColors[i].rgbBlue = i;
        result.bmiColors[i].rgbReserved = 0;
    }

    ////

    returnTrue;
}

//================================================================
//
// writeImage
//
//================================================================

stdbool writeImage
(
    const CharType* basename,
    uint32 id,
    const Point<Space>& imageSize,
    BaseImageProvider& imageProvider,
    ArrayMemory<Pixel>& buffer,
    stdPars(Kit)
)
{
    if_not (imageProvider.dataProcessing())
        returnTrue;

    //
    // File name.
    //

    basic_stringstream<CharType> ss;

    ss << basename;

    if (id != 0)
    {
        ss << CT("-");
        ss << hex << setfill('0') << setw(8) << uppercase <<  id;
    }

    ss << CT(".bmp");

    const CharType* filename = ss.str().c_str();

    //
    //
    //

    Space alignedPitch = 0;
    require(getAlignedPitch(imageSize.X, alignedPitch, stdPass));

    //
    // buffer
    //

    Space tmpSize = 0;
    REQUIRE(safeMul(alignedPitch, imageSize.Y, tmpSize));

    if_not (buffer.resize(tmpSize))
        require(buffer.realloc(tmpSize, imageProvider.baseByteAlignment(), kit.malloc, stdPass));

    ARRAY_EXPOSE(buffer);

    Space bufferMemPitch = alignedPitch;

    //
    // copy image
    //

    Matrix<Pixel> bufferMatrix(bufferPtr, bufferMemPitch, imageSize.X, imageSize.Y);

    REQUIRE(imageProvider.dataProcessing());
    require(imageProvider.saveImage(flipMatrix(bufferMatrix), stdPass));

    returnTrue;
}

//================================================================
//
// BaseConsoleBmpImpl
//
//================================================================

class BaseConsoleBmpImpl
{

public:

    stdbool saveImage(const Matrix<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool setOutputDir(const CharType* outputDir, stdPars(Kit));

private:

    String currentOutputDir;

};

//================================================================
//
// Thunks.
//
//================================================================

BaseConsoleBmp::BaseConsoleBmp()
    {}

BaseConsoleBmp::~BaseConsoleBmp()
    {}

////

stdbool BaseConsoleBmp::saveImage(const Matrix<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(img, desc, id, stdPassThru);}

stdbool BaseConsoleBmp::saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(imageSize, imageProvider, desc, id, stdPassThru);}

////

stdbool BaseConsoleBmp::setOutputDir(const CharType* outputDir, stdPars(Kit))
    {return instance->setOutputDir(outputDir, stdPassThru);}

//================================================================
//
// BaseConsoleBmpImpl::setOutputDir
//
//================================================================

stdbool BaseConsoleBmpImpl::setOutputDir(const CharType* outputDir, stdPars(Kit))
{
    try
    {
        String s = outputDir;

        if (s.length() >= 1)
        {
            auto lastChar = s.substr(s.length() - 1);

            if (lastChar == CT("\\") || lastChar == CT("/"))
                s = s.substr(0, s.length() - 1);
        }

        if (currentOutputDir != s)
        {
            kit.fileTools.makeDirectory(s.c_str());
            currentOutputDir = s;
        }
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("BaseConsoleBmp: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// BaseConsoleBmpImpl::saveImage
//
//================================================================

stdbool BaseConsoleBmpImpl::saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    try
    {
        String descStr;
        require(formatAtomToString(desc, descStr, stdPass));

        String basename = currentOutputDir + CT("/") + mapToFilename(descStr);

        ////

        // require(writer.writeImage(basename.c_str(), id, imageSize, imageProvider, currentFps, currentCodec, currentMaxSegmentFrames, tmpBuffer, stdPass));
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("BaseConsoleBmp: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// BaseConsoleBmpImpl::saveImage
//
//================================================================

stdbool BaseConsoleBmpImpl::saveImage(const Matrix<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    ImageProviderMemcpy imageProvider(image, kit);
    return saveImage(image.size(), imageProvider, desc, id, stdPassThru);
}

//----------------------------------------------------------------

}

