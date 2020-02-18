#include "baseConsoleBmp.h"

#include <sstream>
#include <iomanip>

#include "errorLog/errorLog.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "flipMatrix.h"
#include "formattedOutput/formatStreamStl.h"
#include "data/spacex.h"
#include "baseImageConsole/imageProviderMemcpy.h"
#include "binaryFile/binaryFileImpl.h"

namespace baseConsoleBmp {

using namespace std;

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
// BitmapFileHeader
//
//================================================================

#pragma pack(push, 1)

struct BitmapFileHeader
{
    uint16 bfType;
    uint32 bfSize;
    uint16 bfReserved1;
    uint16 bfReserved2;
    uint32 bfOffBits;
};

COMPILE_ASSERT(sizeof(BitmapFileHeader) == (2*32 + 3*16) / 8);
COMPILE_ASSERT(alignof(BitmapFileHeader) == 1);

//----------------------------------------------------------------

struct BitmapInfoHeader
{
    uint32 biSize;
    int32 biWidth;
    int32 biHeight;
    uint16 biPlanes;
    uint16 biBitCount;
    uint32 biCompression;
    uint32 biSizeImage;
    int32 biXPelsPerMeter;
    int32 biYPelsPerMeter;
    uint32 biClrUsed;
    uint32 biClrImportant;
};

COMPILE_ASSERT(sizeof(BitmapInfoHeader) == (2*16 + 9*32) / 8);
COMPILE_ASSERT(alignof(BitmapInfoHeader) == 1);

//----------------------------------------------------------------

struct BitmapFullHeader : public BitmapFileHeader, public BitmapInfoHeader {};

COMPILE_ASSERT(sizeof(BitmapFullHeader) == sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader));
COMPILE_ASSERT(alignof(BitmapFullHeader) == 1);

//----------------------------------------------------------------

#pragma pack(pop)

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

    //----------------------------------------------------------------
    //
    // File name.
    //
    //----------------------------------------------------------------

    basic_stringstream<CharType> ss;

    ss << basename;

    if (id != 0)
    {
        ss << CT("-");
        ss << hex << setfill('0') << setw(8) << uppercase <<  id;
    }

    ss << CT(".bmp");

    auto str = ss.str();
    auto filename = CharArray{str.data(), str.size()};

    //----------------------------------------------------------------
    //
    // Properly align the buffer.
    //
    //----------------------------------------------------------------

    Space alignedPitch = 0;
    require(getAlignedPitch(imageSize.X, alignedPitch, stdPass));

    ////

    Space imageArea = 0;
    REQUIRE(safeMul(alignedPitch, imageSize.Y, imageArea));

    if_not (buffer.resize(imageArea))
        require(buffer.realloc(imageArea, imageProvider.baseByteAlignment(), kit.malloc, stdPass));

    ARRAY_EXPOSE(buffer);

    Space bufferMemPitch = alignedPitch;

    //----------------------------------------------------------------
    //
    // Copy image to the buffer.
    //
    //----------------------------------------------------------------

    Matrix<Pixel> bufferMatrix(bufferPtr, bufferMemPitch, imageSize.X, imageSize.Y);

    REQUIRE(imageProvider.dataProcessing());
    require(imageProvider.saveImage(flipMatrix(bufferMatrix), stdPass));

    //----------------------------------------------------------------
    //
    // Prepare header.
    //
    //----------------------------------------------------------------

    Space dataSize = bufferSize * sizeof(Pixel);

    Space headerSize = sizeof(BitmapFullHeader);

    Space totalSize{};
    REQUIRE(safeAdd(headerSize, dataSize, totalSize));

    ////

    BitmapFullHeader header;

    ////

    header.bfType = 0x4D42; // BM
    REQUIRE(convertExact(totalSize, header.bfSize));
    header.bfReserved1 = 0; 
    header.bfReserved2 = 0;
    REQUIRE(convertExact(headerSize, header.bfOffBits));

    ////

    header.biSize = sizeof(BitmapInfoHeader);
    REQUIRE(convertExact(imageSize.X, header.biWidth));
    REQUIRE(convertExact(imageSize.Y, header.biHeight));
    header.biPlanes = 1;
    header.biBitCount = sizeof(Pixel) * 8;
    header.biCompression = 0x0000; // BI_RGB
    header.biSizeImage = 0;
    header.biXPelsPerMeter = 3200;
    header.biYPelsPerMeter = 3200;
    header.biClrUsed = 0;
    header.biClrImportant = 0;

    //----------------------------------------------------------------
    //
    // Save to file.
    //
    //----------------------------------------------------------------

    BinaryFileImpl file;
    require(file.open(filename, true, true, stdPass));

    require(file.write(&header, sizeof(header), stdPass));
    require(file.write(bufferPtr, dataSize, stdPass));

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
    ArrayMemory<Pixel> tmpBuffer;

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

        require(writeImage(basename.c_str(), id, imageSize, imageProvider, tmpBuffer, stdPass));
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

