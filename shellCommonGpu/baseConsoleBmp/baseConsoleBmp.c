#include "baseConsoleBmp.h"

#include <unordered_map>

#include "baseInterfaces/imageProviderMemcpy.h"
#include "binaryFile/binaryFileImpl.h"
#include "data/spacex.h"
#include "dataAlloc/arrayMemory.h"
#include "errorLog/errorLog.h"
#include "errorLog/blockExceptions.h"
#include "flipMatrix.h"
#include "formattedOutput/messageFormatterStdio.h"
#include "interfaces/fileTools.h"
#include "rndgen/rndgenBase.h"
#include "storage/rememberCleanup.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "bmpFile/bmpPackedHeaders.h"
#include "cfgTools/boolSwitch.h"
#include "configFile/cfgSimpleString.h"

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

        if (c >= 'A' && c <= 'Z')
            c = c - 'A' + 'a';

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
    MessageFormatterStdio formatter{result};

    v.func(v.value, formatter);
    REQUIRE(formatter.valid());

    REQUIRE(formatter.size() <= size_t{spaceMax});
    result.resize(Space(formatter.size()));

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

    require(imageProvider.saveBgr32(flipMatrix(bufferImage), stdPass));

    //----------------------------------------------------------------
    //
    // Prepare header.
    //
    //----------------------------------------------------------------

    ARRAY_EXPOSE_UNSAFE(bufferArray);

    Space dataSizeInBytes = bufferArraySize * sizeof(Pixel);

    Space headerSize = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);
    COMPILE_ASSERT(sizeof(Pixel) == 4); // Ensure there is no palette

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

    require(file.write(&header, headerSize, stdPass));
    require(file.write(bufferArrayPtr, dataSizeInBytes, stdPass));

    returnTrue;
}

//================================================================
//
// Counter
//
//================================================================

using HashKey = RndgenState;

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
// getDefaultImageDir
//
//================================================================

SimpleString getDefaultImageDir()
{
    SimpleString dir; 
        
    auto tempDir = getenv("HEXLIB_OUTPUT");

    if_not (tempDir)
        tempDir = getenv("TEMP");

    if (tempDir)
        dir << tempDir << "/imageConsole";

    if_not (def(dir))
        dir.clear();

    return dir;
}

//================================================================
//
// BaseConsoleBmpImpl
//
//================================================================

class BaseConsoleBmpImpl : public BaseConsoleBmp
{

public:

    void setActive(bool active)
        {savingActive = active;}

    void setDir(const CharType* dir)
        {outputDir = dir ? SimpleString(dir) : getDefaultImageDir();}

public:

    void serialize(const CfgSerializeKit& kit)
    {
        savingActive.serialize(kit, STR("Active"), STR("Shift+Alt+B"));
        outputDir.serialize(kit, STR("Output Directory"));
    }

public:

    stdbool saveImage(const Matrix<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

public:

    bool active() const
        {return savingActive;}

    const CharType* getOutputDir() const
        {return outputDir().cstr();}

    void setLockstepCounter(Counter counter);

private:

    BoolSwitch<false> savingActive;
    SimpleStringVar outputDir{getDefaultImageDir()};

private:

    Counter initCounter = 0;
    unordered_map<HashKey, Counter> table;

private:

    stdbool mdCheck(stdPars(Kit));

    SimpleString mdCurrentDir = nanOf<SimpleString>();

};

//----------------------------------------------------------------

UniquePtr<BaseConsoleBmp> BaseConsoleBmp::create()
    {return makeUnique<BaseConsoleBmpImpl>();}

//================================================================
//
// BaseConsoleBmpImpl::setLockstepCounter
//
//================================================================

void BaseConsoleBmpImpl::setLockstepCounter(Counter counter)
{
    initCounter = counter;

    for (auto& value: table)
        value.second = counter;
}

//================================================================
//
// BaseConsoleBmpImpl::mdCheck
//
//================================================================

stdbool BaseConsoleBmpImpl::mdCheck(stdPars(Kit))
{
    auto& dir = outputDir();
    REQUIRE(def(dir));

    if_not (mdCurrentDir == dir)
    {
        mdCurrentDir = dir;
        REQUIRE(def(mdCurrentDir));

        kit.fileTools.makeDirectory(mdCurrentDir.cstr()); // don't check for error
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
    require(mdCheck(stdPass));

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
    require(bufferArray.realloc(sizeInPixels, imageProvider.desiredBaseByteAlignment(), stdPass));

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
        auto f = table.insert(make_pair(hash, initCounter));
        Counter& counter = f.first->second;
        frameIndex = counter++;
    };

    require(blockExceptionsVoid(getFrameIndex()));

    //----------------------------------------------------------------
    //
    // Format file name.
    //
    //----------------------------------------------------------------

    auto filenameMsg = paramMsg(STR("%/%--%.bmp"), outputDir(), dec(frameIndex, 8), descArray());

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
    require(mdCheck(stdPass));

    ImageProviderMemcpy imageProvider(image, kit);
    return saveImage(image.size(), imageProvider, desc, id, stdPassThru);
}

//----------------------------------------------------------------

}

