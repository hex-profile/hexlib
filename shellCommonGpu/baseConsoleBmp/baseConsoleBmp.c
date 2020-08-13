#include "baseConsoleBmp.h"

#include <unordered_map>

#include "baseImageConsole/imageProviderMemcpy.h"
#include "binaryFile/binaryFileImpl.h"
#include "data/spacex.h"
#include "dataAlloc/arrayMemory.h"
#include "errorLog/errorLog.h"
#include "errorLog/convertAllExceptions.h"
#include "flipMatrix.h"
#include "formattedOutput/formatStreamStdio.h"
#include "interfaces/fileTools.h"
#include "rndgen/rndgenBase.h"
#include "storage/rememberCleanup.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"

namespace baseConsoleBmp {

using namespace std;

//================================================================
//
// fixFilename
//
//================================================================

stdbool fixFilename(const Array<const CharType>& src, const Array<CharType>& dst, stdPars(Kit))
{
    REQUIRE(equalSize(src, dst));
    ARRAY_EXPOSE(src);
    ARRAY_EXPOSE(dst);

    ////

    for_count (i, dstSize)
    {
        CharType c = srcPtr[i];

        if_not
        (
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '.' || c == '-'
        )
            c = '-';

        dstPtr[i] = c;
    }

    returnTrue;
}

//================================================================
//
// formatAtomToBuffer
//
//================================================================

stdbool formatAtomToBuffer(const FormatOutputAtom& v, ArrayMemory<CharType>& result, stdPars(ErrorLogKit))
{
    ARRAY_EXPOSE_UNSAFE(result);
    FormatStreamStdioThunk formatter(resultPtr, resultSize);

    v.func(v.value, formatter);
    REQUIRE(formatter.valid());

    REQUIRE(formatter.usedSize() <= size_t{spaceMax});
    result.resize(Space(formatter.usedSize()));

    returnTrue;
}

//================================================================
//
// bmpAlignment
//
//================================================================

static const Space bmpAlignmentMask = 3;

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
    const CharArray& filename,
    uint32 id,
    const Point<Space>& imageSize,
    BaseImageProvider& imageProvider,
    const Matrix<Pixel>& bufferImage,
    const Array<Pixel>& bufferArray,
    stdPars(Kit)
)
{
    //----------------------------------------------------------------
    //
    // Copy image to the buffer.
    //
    //----------------------------------------------------------------

    REQUIRE(kit.dataProcessing);

    require(imageProvider.saveImage(flipMatrix(bufferImage), stdPass));

    //----------------------------------------------------------------
    //
    // Prepare header.
    //
    //----------------------------------------------------------------

    ARRAY_EXPOSE_UNSAFE(bufferArray);

    Space dataSizeInBytes = bufferArraySize * sizeof(Pixel);

    Space headerSize = sizeof(BitmapFullHeader);

    Space totalSize{};
    REQUIRE(safeAdd(headerSize, dataSizeInBytes, totalSize));

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
    require(file.write(bufferArrayPtr, dataSizeInBytes, stdPass));

    returnTrue;
}

//================================================================
//
// Counter
//
//================================================================

using HashKey = RndgenState;
using Counter = uint32;

//================================================================
//
// getHash
//
//================================================================

HashKey getHash(const CharArray& str)
{
    RndgenState result = 0;

    for_count (i, str.size)
    {
        result += str.ptr[i];
        rndgenNext(result);
    }

    return result;
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

    SimpleString currentOutputDir;
    
    unordered_map<HashKey, Counter> table;

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
    if_not (currentOutputDir == outputDir)
    {
        kit.fileTools.makeDirectory(outputDir);
        currentOutputDir = outputDir;
        REQUIRE(def(currentOutputDir));
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

    //----------------------------------------------------------------
    //
    // Format buffer.
    //
    //----------------------------------------------------------------

    ARRAY_ALLOC(descArray, CharType, 8192);
    ARRAY_ALLOC(filenameArray, CharType, 8192);

    //----------------------------------------------------------------
    //
    // Buffer image.
    //
    //----------------------------------------------------------------

    Space alignedPitch = 0;
    require(getAlignedPitch(imageSize.X, alignedPitch, stdPass));

    Space sizeInPixels = 0;
    REQUIRE(safeMul(alignedPitch, imageSize.Y, sizeInPixels));

    ArrayMemory<Pixel> bufferArray;
    require(bufferArray.realloc(sizeInPixels, imageProvider.baseByteAlignment(), stdPass));

    ARRAY_EXPOSE(bufferArray);

    Matrix<Pixel> bufferImage(bufferArrayPtr, alignedPitch, imageSize.X, imageSize.Y);

    //----------------------------------------------------------------
    //
    // Everything is allocated, exit on counting stage.
    //
    //----------------------------------------------------------------

    if_not (kit.dataProcessing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Format description.
    //
    //----------------------------------------------------------------

    auto descWithID = (id == 0) ? desc : paramMsg(STR("%-%"), desc, hex(id, 8));

    require(formatAtomToBuffer(descWithID, descArray, stdPass));

    require(fixFilename(descArray, descArray, stdPass));

    ARRAY_EXPOSE_UNSAFE(descArray);
    CharArray descStr(descArrayPtr, descArraySize);

    //----------------------------------------------------------------
    //
    // Access counters.
    //
    //----------------------------------------------------------------

    Counter frameIndex{};

    auto getFrameIndex = [&]
    {
        auto hash = getHash(descStr);
        auto f = table.insert(make_pair(hash, Counter{}));
        Counter& counter = f.first->second;
        frameIndex = counter++;
        returnTrue;
    };

    require(convertAllExceptions(getFrameIndex()));

    //----------------------------------------------------------------
    //
    // Format file name.
    //
    //----------------------------------------------------------------

    auto filenameMsg = paramMsg(STR("%/%-%.bmp"), currentOutputDir, descArray(), dec(frameIndex, 8));

    require(formatAtomToBuffer(filenameMsg, filenameArray, stdPass));

    ARRAY_EXPOSE_UNSAFE(filenameArray);
    CharArray filenameStr(filenameArrayPtr, filenameArraySize);

    //----------------------------------------------------------------
    //
    // Write image.
    //
    //----------------------------------------------------------------

    require(writeImage(filenameStr, id, imageSize, imageProvider, bufferImage, bufferArray, stdPass));

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

