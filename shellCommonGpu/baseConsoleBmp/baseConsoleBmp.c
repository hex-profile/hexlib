#include "baseConsoleBmp.h"

#include <unordered_map>

#include "baseInterfaces/imageProviderMemcpy.h"
#include "binaryFile/binaryFileImpl.h"
#include "bmpFile/bmpPackedHeaders.h"
#include "cfgTools/boolSwitch.h"
#include "cfgTools/cfgSimpleString.h"
#include "data/spacex.h"
#include "dataAlloc/arrayMemory.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"
#include "errorLog/errorLog.h"
#include "flipMatrix.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
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

void fixFilename(const Array<const CharType>& src, const Array<CharType>& dst, stdPars(Kit))
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
}

//================================================================
//
// formatAtomToBuffer
//
//================================================================

void formatAtomToBuffer(const FormatOutputAtom& v, ArrayMemory<CharType>& result, stdPars(ErrorLogKit))
{
    ARRAY_EXPOSE_UNSAFE(result);
    MessageFormatterImpl formatter{result};

    v.func(v.value, formatter);
    REQUIRE(formatter.valid());

    REQUIRE(formatter.size() <= size_t{spaceMax});
    result.resize(Space(formatter.size()));
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

void getAlignedPitch(Space sizeX, Space& pitch, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0);

    REQUIRE(sizeX <= spaceMax / Space(sizeof(Pixel)));
    Space rowMemSize = sizeX * Space(sizeof(Pixel));

    REQUIRE(rowMemSize <= spaceMax - bmpAlignmentMask);
    Space rowAlignedSize = (rowMemSize + bmpAlignmentMask) & (~bmpAlignmentMask);

    Space bufSizeX = rowAlignedSize / Space(sizeof(Pixel));
    REQUIRE(bufSizeX * Space(sizeof(Pixel)) == rowAlignedSize);

    pitch = bufSizeX;
}

//================================================================
//
// writeImage
//
//================================================================

void writeImage
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

    imageProvider.saveBgr32(flipMatrix(bufferImage), stdPass);

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

    BitmapHeaderWithPalette header;

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
    file.open(filename, true, true, stdPass);

    file.write(&header, headerSize, stdPass);
    file.write(bufferArrayPtr, dataSizeInBytes, stdPass);
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

    void setActive(bool active)
        {savingActive = active;}

    void setDir(const CharType* dir)
        {outputDir = dir ? SimpleString(dir) : getDefaultImageDir();}

    ////

    void serialize(const CfgSerializeKit& kit, bool hotkeys)
    {
        savingActive.serialize(kit, STR("Active"), hotkeys ? STR("Shift+Alt+B") : STR(""));
        outputDir.serialize(kit, STR("Output Directory"));
    }

    ////

    void saveImage(const MatrixAP<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    void saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    ////

    void clearState();

    bool active() const
        {return savingActive;}

    const CharType* getOutputDir() const
        {return outputDir().cstr();}

    void setLockstepCounter(Counter counter);

    ////

    BoolSwitch savingActive{false};
    SimpleStringVar outputDir{getDefaultImageDir()};

    ////

    Counter initCounter = 0;
    unordered_map<HashKey, Counter> table;

    ////

    void mdCheck(stdPars(Kit));

    SimpleString mdCurrentDir = nanOf<SimpleString>();

};

//----------------------------------------------------------------

UniquePtr<BaseConsoleBmp> BaseConsoleBmp::create()
    {return makeUnique<BaseConsoleBmpImpl>();}

//================================================================
//
// BaseConsoleBmpImpl::clearState
//
//================================================================

void BaseConsoleBmpImpl::clearState()
{
    initCounter = 0;
    table.clear();
    mdCurrentDir = nanOf<SimpleString>();
}

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

void BaseConsoleBmpImpl::mdCheck(stdPars(Kit))
{
    auto& dir = outputDir();
    REQUIRE(def(dir));

    if_not (mdCurrentDir == dir)
    {
        mdCurrentDir = dir;
        REQUIRE(def(mdCurrentDir));

        fileTools::makeDirectory(mdCurrentDir.cstr()); // don't check for error
    }
}

//================================================================
//
// BaseConsoleBmpImpl::saveImage
//
//================================================================

void BaseConsoleBmpImpl::saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    mdCheck(stdPass);

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
    getAlignedPitch(imageSize.X, alignedPitch, stdPass);

    Space sizeInPixels = 0;
    REQUIRE(safeMul(alignedPitch, imageSize.Y, sizeInPixels));

    ArrayMemory<Pixel> bufferArray;
    bufferArray.realloc(sizeInPixels, imageProvider.desiredBaseByteAlignment(), stdPass);

    ARRAY_EXPOSE(bufferArray);

    Matrix<Pixel> bufferImage;
    bufferImage.assignUnsafe(bufferArrayPtr, alignedPitch, imageSize.X, imageSize.Y);

    //----------------------------------------------------------------
    //
    // Everything is allocated, exit on counting stage.
    //
    //----------------------------------------------------------------

    if_not (kit.dataProcessing)
        return;

    //----------------------------------------------------------------
    //
    // Format description.
    //
    //----------------------------------------------------------------

    auto descWithID = (id == 0) ? desc : paramMsg(STR("%-%"), desc, hex(id, 8));

    formatAtomToBuffer(descWithID, descArray, stdPass);

    fixFilename(descArray, descArray, stdPass);

    ARRAY_EXPOSE_UNSAFE(descArray);
    CharArray descStr(descArrayPtr, descArraySize);

    //----------------------------------------------------------------
    //
    // Access counters.
    //
    //----------------------------------------------------------------

    Counter frameIndex{};

    {
        convertExceptionsBegin;

        auto hash = getHash(descStr);
        auto f = table.insert(make_pair(hash, initCounter));
        Counter& counter = f.first->second;
        frameIndex = counter++;

        convertExceptionsEnd;
    }

    //----------------------------------------------------------------
    //
    // Format file name.
    //
    //----------------------------------------------------------------

    formatAtomToBuffer(paramMsg(STR("%/%--%.bmp"), outputDir(), dec(frameIndex, 8), descArray()), filenameArray, stdPass);

    ARRAY_EXPOSE_UNSAFE(filenameArray);
    CharArray filenameStr(filenameArrayPtr, filenameArraySize);

    //----------------------------------------------------------------
    //
    // Write image.
    //
    //----------------------------------------------------------------

    writeImage(filenameStr, id, imageSize, imageProvider, bufferImage, bufferArray, stdPass);
}

//================================================================
//
// BaseConsoleBmpImpl::saveImage
//
//================================================================

void BaseConsoleBmpImpl::saveImage(const MatrixAP<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    mdCheck(stdPass);

    ImageProviderMemcpy imageProvider(image, kit);
    saveImage(image.size(), imageProvider, desc, id, stdPassThru);
}

//----------------------------------------------------------------

}

