#include "baseConsoleAvi.h"

#include <string>
#include <map>
#include <sstream>
#include <iomanip>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vfw.h>

#include "errorLog/errorLog.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "flipMatrix.h"
#include "formattedOutput/messageFormatterStdio.h"
#include "data/spacex.h"
#include "baseImageConsole/imageProviderMemcpy.h"
#include "userOutput/paramMsg.h"
#include "interfaces/fileTools.h"

namespace baseConsoleAvi {

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
// codecFromStr
//
//================================================================

Codec codecFromStr(const CharType* s)
{
    if_not (strlen(s) == 4)
        return 0;

    return MAKEFOURCC(s[0], s[1], s[2], s[3]);
}

//================================================================
//
// String
//
//================================================================

using String = basic_string<CharType>;

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

    for_count (i, 256)
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
// AviFile
//
//================================================================

class AviFile
{

public:

    AviFile() : handle(0) {}
    ~AviFile() {close();}

    bool open(const CharType* filename)
    {
        close();
        int r = AVIFileOpen(&handle, filename, OF_CREATE | OF_WRITE, NULL);
        return r == 0;
    }

    void close()
    {
        if (handle != 0)
        {
            AVIFileRelease(handle);
            handle = 0;
        }
    }

    operator PAVIFILE () const {return handle;}
    PAVIFILE operator () () const {return handle;}

private:

    PAVIFILE handle;

};

//================================================================
//
// AviStream
//
//================================================================

class AviStream
{

public:

    AviStream() : stream(0) {}
    ~AviStream() {close();}

    bool create(PAVIFILE avifile, AVISTREAMINFO* info)
    {
        close();
        return AVIFileCreateStream(avifile, &stream, info) == 0;
    }

    void close()
    {
        if (stream != 0)
        {
            AVIStreamRelease(stream);
            stream = 0;
        }
    }

    operator PAVISTREAM () const {return stream;}
    PAVISTREAM operator () () const {return stream;}

private:

    PAVISTREAM stream;

};

//================================================================
//
// AviCompressedStream
//
//================================================================

class AviCompressedStream
{

public:

    AviCompressedStream() : stream(0) {}
    ~AviCompressedStream() {close();}

    bool create(PAVISTREAM baseStream, AVICOMPRESSOPTIONS* options)
    {
        close();
        return AVIMakeCompressedStream(&stream, baseStream, options, NULL) == 0;
    }

    void close()
    {
        if (stream != 0)
        {
            AVIStreamRelease(stream);
            stream = 0;
        }
    }

    operator PAVISTREAM () const {return stream;}
    PAVISTREAM operator () () const {return stream;}

private:

    PAVISTREAM stream;

};

//================================================================
//
// AviWriter
//
//================================================================

class AviWriter
{

public:

    AviWriter()
        :
        currentSize(point(0)),
        lastSize(point(0))
    {
    }

    stdbool writeImage
    (
        const CharType* basename,
        uint32 id,
        const Point<Space>& imageSize,
        BaseImageProvider& imageProvider,
        FPS fps,
        Codec codec,
        int32 maxSegmentFrames,
        const Matrix<Pixel>& bufferImage,
        const Array<Pixel>& bufferArray,
        stdPars(Kit)
    );

private:

    stdbool open(const CharType* filename, const Point<Space>& size, FPS fps, Codec codec, stdPars(Kit));

private:

    void close()
    {
        aviStreamCompressed.close();
        aviStreamBase.close();
        aviFile.close();

        currentPosition = 0;
        currentSize = point(0);
        currentFps = 0;
        currentCodec = 0;
    }

private:
    
    bool opened() const {return aviFile && aviStreamBase && aviStreamCompressed;}

private:

    AviFile aviFile;
    AviStream aviStreamBase;
    AviCompressedStream aviStreamCompressed;

    String currentBasename;
    int32 currentPosition = 0;
    Point<Space> currentSize;
    FPS currentFps = 0;
    Codec currentCodec = 0;

    int32 segmentNumber = 0;

    //
    // entrance cache
    //

    String lastBasename;
    Point<Space> lastSize;
    FPS lastFps = 0;
    Codec lastCodec = 0;
    bool lastResult = false;

};

//================================================================
//
// AviWriter::open
//
//================================================================

stdbool AviWriter::open(const CharType* filename, const Point<Space>& size, FPS fps, Codec codec, stdPars(Kit))
{
    //
    // close
    //

    close();

    ////

    if_not (aviFile.open(filename))
    {
        printMsg(kit.msgLog, STR("Cannot write file %0"), filename, msgErr);
        returnFalse;
    }

    REMEMBER_CLEANUP1_EX(aviFileClose, aviFile.close(), AviFile&, aviFile);

    ////

    AVISTREAMINFO info;
    memset(&info, 0, sizeof(info));

    info.fccType = streamtypeVIDEO;
    info.fccHandler = MAKEFOURCC('v', 'i', 'd', 's');
    ASSIGN_CONVERT(info.dwRate, fps);
    info.dwScale = 1;
    info.dwQuality = -1;

    ////

    REQUIRE(aviStreamBase.create(aviFile, &info));
    REMEMBER_CLEANUP1_EX(aviStreamBaseClose, aviStreamBase.close(), AviStream&, aviStreamBase);

    //
    // create compressed stream
    //

    BitmapinfoPalette format;
    memset(&format, 0, sizeof(format));
    require(makeBitmapHeader<Pixel>(size, format, stdPass));

    ////

    AVICOMPRESSOPTIONS options;
    memset(&options, 0, sizeof(options));

    options.fccType = streamtypeVIDEO;
    options.fccHandler = codec;

    if_not (aviStreamCompressed.create(aviStreamBase, &options))
    {
        printMsg(kit.msgLog, STR("Cannot create compressed AVI stream for %0"), filename, msgErr);
        returnFalse;
    }

    ////

    if_not (AVIStreamSetFormat(aviStreamCompressed, 0, &format, sizeof(format)) == 0)
    {
        printMsg(kit.msgLog, STR("Cannot set compressed AVI stream for %s"), filename, msgErr);
        returnFalse;
    }

    //
    // record success
    //

    currentPosition = 0;
    currentSize = size;
    currentFps = fps;
    currentCodec = codec;

    aviFileClose.cancel();
    aviStreamBaseClose.cancel();

    ////

    returnTrue;
}

//================================================================
//
// AviWriter::writeImage
//
//================================================================

stdbool AviWriter::writeImage
(
    const CharType* basename,
    uint32 id,
    const Point<Space>& imageSize,
    BaseImageProvider& imageProvider,
    FPS fps,
    Codec codec,
    int32 maxSegmentFrames,
    const Matrix<Pixel>& bufferImage,
    const Array<Pixel>& bufferArray,
    stdPars(Kit)
)
{

    REQUIRE(maxSegmentFrames >= 0);

    REQUIRE(kit.dataProcessing);

    //----------------------------------------------------------------
    //
    // If have already tried it and failed, don't issue duplicate errors.
    //
    //----------------------------------------------------------------

    if
    (
        basename == lastBasename &&
        allv(imageSize == lastSize) &&
        fps == lastFps &&
        codec == lastCodec &&
        lastResult == false
    )
        returnFalse;

    lastBasename = basename;
    lastSize = imageSize;
    lastFps = fps;
    lastCodec = codec;
    lastResult = false;

    //----------------------------------------------------------------
    //
    // Reopen if neccessary.
    //
    //----------------------------------------------------------------

    if_not
    (
        opened() &&
        allv(imageSize == currentSize) &&
        fps == currentFps &&
        codec == currentCodec &&
        (!maxSegmentFrames || currentPosition < maxSegmentFrames)
    )
    {
        if (aviFile != 0)
            ++segmentNumber;

        basic_stringstream<CharType> ss;

        if (currentBasename.empty())
            currentBasename = basename;

        ss << currentBasename;

        if (id != 0)
        {
            ss << CT("-");
            ss << hex << setfill('0') << setw(8) << uppercase <<  id;
        }

        if (segmentNumber != 0)
            ss << CT("-") << dec << setw(4) << setfill(CT('0')) << segmentNumber;

        ss << CT(".avi");

        auto str = ss.str();
        require(open(str.c_str(), imageSize, fps, codec, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Copy image.
    //
    //----------------------------------------------------------------

    REQUIRE(kit.dataProcessing);
    require(imageProvider.saveImage(flipMatrix(bufferImage), stdPass));

    //----------------------------------------------------------------
    //
    // Write.
    //
    //----------------------------------------------------------------

    ARRAY_EXPOSE(bufferArray);

    REQUIRE
    (
        AVIStreamWrite
        (
            aviStreamCompressed,
            currentPosition, 1,
            unsafePtr(bufferArrayPtr, bufferArraySize),
            bufferArraySize * sizeof(Pixel),
            AVIIF_KEYFRAME,
            NULL, NULL
        )
        == 0
    );

    //----------------------------------------------------------------
    //
    // Record success.
    //
    //----------------------------------------------------------------

    ++currentPosition;

    lastResult = true;

    returnTrue;
}

//================================================================
//
// FileId
//
//================================================================

struct FileId
{
    String name;
    uint32 id;

    FileId(const String& name, uint32 id)
        : name(name), id(id) {}
};

//----------------------------------------------------------------

inline bool operator <(const FileId& A, const FileId& B)
{
    if (A.id == 0 && B.id == 0)
        return (A.name < B.name);
    else
        return (A.id < B.id);
}

//================================================================
//
// BaseConsoleAviImpl
//
//================================================================

class BaseConsoleAviImpl
{

public:

    BaseConsoleAviImpl() {CoInitialize(0);} // for VFW

    stdbool saveImage(const Matrix<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool setOutputDir(const CharType* outputDir, stdPars(Kit));
    stdbool setFps(FPS fps, stdPars(Kit));
    stdbool setCodec(Codec codec, stdPars(Kit)) {currentCodec = codec; returnTrue;}
    stdbool setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit)) {currentMaxSegmentFrames = maxSegmentFrames; returnTrue;}

private:

    String currentOutputDir;
    FPS currentFps = 60;
    Codec currentCodec = MAKEFOURCC('D', 'I', 'B', ' ');
    int32 currentMaxSegmentFrames = 0;

    using WritersMap = map<FileId, AviWriter>;
    WritersMap writers;

};

//================================================================
//
// Thunks.
//
//================================================================

BaseConsoleAvi::BaseConsoleAvi()
    {}

BaseConsoleAvi::~BaseConsoleAvi()
    {}

////

stdbool BaseConsoleAvi::saveImage(const Matrix<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(img, desc, id, stdPassThru);}

stdbool BaseConsoleAvi::saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(imageSize, imageProvider, desc, id, stdPassThru);}

////

stdbool BaseConsoleAvi::setOutputDir(const CharType* outputDir, stdPars(Kit))
    {return instance->setOutputDir(outputDir, stdPassThru);}

stdbool BaseConsoleAvi::setFps(const FPS& fps, stdPars(Kit))
    {return instance->setFps(fps, stdPassThru);}

stdbool BaseConsoleAvi::setCodec(const Codec& codec, stdPars(Kit))
    {return instance->setCodec(codec, stdPassThru);}

stdbool BaseConsoleAvi::setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit))
    {return instance->setMaxSegmentFrames(maxSegmentFrames, stdPassThru);}

//================================================================
//
// BaseConsoleAviImpl::setOutputDir
//
//================================================================

stdbool BaseConsoleAviImpl::setOutputDir(const CharType* outputDir, stdPars(Kit))
{
    try
    {
        if (currentOutputDir != outputDir)
        {
            kit.fileTools.makeDirectory(outputDir);
            writers.clear();
            currentOutputDir = outputDir;
        }
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("BaseConsoleAvi: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// BaseConsoleAviImpl::setOutputDir
//
//================================================================

stdbool BaseConsoleAviImpl::setFps(FPS fps, stdPars(Kit))
{
    REQUIRE(fps >= 1 && fps <= 1024);
    currentFps = fps;

    returnTrue;
}

//================================================================
//
// BaseConsoleAviImpl::saveImage
//
//================================================================

stdbool BaseConsoleAviImpl::saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    try
    {
        //----------------------------------------------------------------
        //
        // Format buffer.
        //
        //----------------------------------------------------------------

        ARRAY_ALLOC(descArray, CharType, 8192);
        ARRAY_ALLOC(basenameArray, CharType, 8192);

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
        // Format base name.
        //
        //----------------------------------------------------------------

        require(formatAtomToBuffer(desc, descArray, stdPass));

        require(fixFilename(descArray, descArray, stdPass));

        //----------------------------------------------------------------
        //
        // Proceed.
        //
        //----------------------------------------------------------------

        auto basenameMsg = paramMsg(STR("%/%"), currentOutputDir.c_str(), descArray());
        require(formatAtomToBuffer(basenameMsg, basenameArray, stdPass));

        ARRAY_EXPOSE_UNSAFE(basenameArray);
        String basenameStr(basenameArrayPtr, basenameArraySize);

        ////

        auto f = writers.insert(make_pair(FileId(basenameStr, id), AviWriter{}));
        AviWriter& writer = f.first->second;
        require(writer.writeImage(basenameStr.c_str(), id, imageSize, imageProvider, currentFps, currentCodec, currentMaxSegmentFrames, bufferImage, bufferArray, stdPass));
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("BaseConsoleAvi: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// BaseConsoleAviImpl::saveImage
//
//================================================================

stdbool BaseConsoleAviImpl::saveImage(const Matrix<const Pixel>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    ImageProviderMemcpy imageProvider(image, kit);
    return saveImage(image.size(), imageProvider, desc, id, stdPassThru);
}

//----------------------------------------------------------------

}
