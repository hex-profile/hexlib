#include "outImgAvi.h"

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
#include "formattedOutput/formatStreamStl.h"
#include "data/spacex.h"

namespace outImgAvi {

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

template <typename Element>
stdbool getAlignedPitch(Space sizeX, Space& pitch, stdPars(ErrorLogKit))
{
    REQUIRE(sizeX >= 0);

    REQUIRE(sizeX <= spaceMax / Space(sizeof(Element)));
    Space rowMemSize = sizeX * Space(sizeof(Element));

    REQUIRE(rowMemSize <= spaceMax - bmpAlignmentMask);
    Space rowAlignedSize = (rowMemSize + bmpAlignmentMask) & (~bmpAlignmentMask);

    Space bufSizeX = rowAlignedSize / Space(sizeof(Element));
    REQUIRE(bufSizeX * Space(sizeof(Element)) == rowAlignedSize);

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

template <typename Element>
stdbool makeBitmapHeader(const Point<Space>& size, BitmapinfoPalette& result, stdPars(ErrorLogKit))
{
    BITMAPINFOHEADER& bmi = result.bmiHeader;

    ////

    REQUIRE(size.X >= 0 && size.Y >= 0);

    //
    // check dimensions
    //

    Space alignedPitch = 0;
    require(getAlignedPitch<Element>(size.X, alignedPitch, stdPass));

    //
    // fill the structure
    //

    bmi.biSize = sizeof(BITMAPINFOHEADER);
    ASSIGN_CONVERT(bmi.biWidth, size.X);
    ASSIGN_CONVERT(bmi.biHeight, size.Y);
    bmi.biPlanes = 1;
    bmi.biBitCount = sizeof(Element) * 8;
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

    KIT_COMBINE3(Kit, ErrorLogKit, MsgLogsKit, MallocKit);

public:

    AviWriter()
        :
        currentSize(point(0)),
        lastSize(point(0))
    {
    }

    template <typename Element>
    stdbool writeImage
    (
        const CharType* basename,
        uint32 id,
        const Point<Space>& imageSize,
        AtImageProvider<Element>& imageProvider,
        FPS fps,
        Codec codec,
        int32 maxSegmentFrames,
        ArrayMemory<Element>& buffer,
        stdPars(Kit)
    );

private:

    template <typename Element>
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

template <typename Element>
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
    require(makeBitmapHeader<Element>(size, format, stdPass));

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

template <typename Element>
stdbool AviWriter::writeImage
(
    const CharType* basename,
    uint32 id,
    const Point<Space>& imageSize,
    AtImageProvider<Element>& imageProvider,
    FPS fps,
    Codec codec,
    int32 maxSegmentFrames,
    ArrayMemory<Element>& buffer,
    stdPars(Kit)
)
{
    if_not (imageProvider.dataProcessing())
        returnTrue;

    REQUIRE(maxSegmentFrames >= 0);

    //
    // if have already tried it and failed, don't issue duplicate errors
    //

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

    //
    // reopen if neccessary
    //

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

        require(open<Element>(ss.str().c_str(), imageSize, fps, codec, stdPass));
    }

    //
    //
    //

    Space alignedPitch = 0;
    require(getAlignedPitch<Element>(imageSize.X, alignedPitch, stdPass));

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

    Matrix<Element> bufferMatrix(bufferPtr, bufferMemPitch, imageSize.X, imageSize.Y);

    REQUIRE(imageProvider.dataProcessing());
    require(imageProvider.saveImage(flipMatrix(bufferMatrix), stdPass));

    //
    // write
    //

    REQUIRE
    (
        AVIStreamWrite
        (
            aviStreamCompressed,
            currentPosition, 1,
            unsafePtr(bufferPtr, bufferSize),
            bufferSize * sizeof(Element),
            AVIIF_KEYFRAME,
            NULL, NULL
        )
        == 0
    );

    //
    // record success
    //

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
// OutImgAviImpl
//
//================================================================

class OutImgAviImpl
{

    using Kit = OutImgAvi::Kit;

public:

    OutImgAviImpl() {CoInitialize(0);} // for VFW

    template <typename Element>
    stdbool saveImageGeneric(const Point<Space>& imageSize, AtImageProvider<Element>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    template <typename Element>
    stdbool saveImage(const Matrix<const Element>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool saveImage(const Point<Space>& imageSize, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

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

private:

    ArrayMemory<uint8> tmpBuffer8;
    ArrayMemory<uint8_x4> tmpBuffer32;

    template <typename Element>
    ArrayMemory<Element>& getTmpBuffer();

    template <>
    ArrayMemory<uint8>& getTmpBuffer() {return tmpBuffer8;}

    template <>
    ArrayMemory<uint8_x4>& getTmpBuffer() {return tmpBuffer32;}

};

//================================================================
//
// thunks
//
//================================================================

OutImgAvi::OutImgAvi()
    {}

OutImgAvi::~OutImgAvi()
    {}

////

stdbool OutImgAvi::saveImage(const Matrix<const uint8>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(img, desc, id, stdPassThru);}

stdbool OutImgAvi::saveImage(const Matrix<const uint8_x4>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(img, desc, id, stdPassThru);}

stdbool OutImgAvi::saveImage(const Point<Space>& imageSize, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
    {return instance->saveImage(imageSize, imageProvider, desc, id, stdPassThru);}

////

stdbool OutImgAvi::setOutputDir(const CharType* outputDir, stdPars(Kit))
    {return instance->setOutputDir(outputDir, stdPassThru);}

stdbool OutImgAvi::setFps(const FPS& fps, stdPars(Kit))
    {return instance->setFps(fps, stdPassThru);}

stdbool OutImgAvi::setCodec(const Codec& codec, stdPars(Kit))
    {return instance->setCodec(codec, stdPassThru);}

stdbool OutImgAvi::setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit))
    {return instance->setMaxSegmentFrames(maxSegmentFrames, stdPassThru);}

//================================================================
//
// OutImgAviImpl::setOutputDir
//
//================================================================

stdbool OutImgAviImpl::setOutputDir(const CharType* outputDir, stdPars(Kit))
{
    try
    {
        String s = outputDir;

        if (s.length() >= 1 && s.substr(s.length() - 1) == CT("\\"))
            s = s.substr(0, s.length() - 1);

        if (currentOutputDir != s)
        {
            kit.fileTools.makeDirectory(s.c_str());
            writers.clear();
            currentOutputDir = s;
        }
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("OutImgAvi: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// OutImgAviImpl::setOutputDir
//
//================================================================

stdbool OutImgAviImpl::setFps(FPS fps, stdPars(Kit))
{
    REQUIRE(fps >= 1 && fps <= 1024);
    currentFps = fps;

    returnTrue;
}

//================================================================
//
// ImageProviderMemcpy
//
//================================================================

template <typename Element>
class ImageProviderMemcpy : public AtImageProvider<Element>
{

public:

    ImageProviderMemcpy(const Matrix<const Element>& source, const ErrorLogKit& kit)
        : source(source), kit(kit) {}

public:

    bool dataProcessing() const
        {return true;}

    Space getPitch() const
        {return source.memPitch();}

    Space baseByteAlignment() const
        {return cpuBaseByteAlignment;}

    stdbool saveImage(const Matrix<Element>& dest, stdNullPars);

private:

    Matrix<const Element> source;
    ErrorLogKit kit;

};

//================================================================
//
// ImageProviderMemcpy::saveImage
//
//================================================================

template <typename Element>
stdbool ImageProviderMemcpy<Element>::saveImage(const Matrix<Element>& dest, stdNullPars)
{
    REQUIRE(source.size() == dest.size());

    MATRIX_EXPOSE(source);
    MATRIX_EXPOSE(dest);

    MatrixPtr(const Element) sourceRow = sourceMemPtr;
    MatrixPtr(Element) destRow = destMemPtr;

    for (Space Y = 0; Y < sourceSizeY; ++Y)
    {
        memcpy(unsafePtr(destRow, sourceSizeX), unsafePtr(sourceRow, sourceSizeX), sourceSizeX * sizeof(Element));
        destRow += destMemPitch;
        sourceRow += sourceMemPitch;
    }

    returnTrue;
}

//================================================================
//
// OutImgAviImpl::saveImage
//
//================================================================

template <typename Element>
stdbool OutImgAviImpl::saveImageGeneric(const Point<Space>& imageSize, AtImageProvider<Element>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    ArrayMemory<Element>& tmpBuffer = getTmpBuffer<Element>();

    try
    {
        String descStr;
        require(formatAtomToString(desc, descStr, stdPass));

        String basename = currentOutputDir + CT("\\") + mapToFilename(descStr);

        ////

        using MapIterator = WritersMap::iterator;
        pair<MapIterator, bool> f = writers.insert(make_pair(FileId(basename, id), AviWriter()));
        AviWriter& writer = f.first->second;
        require(writer.writeImage(basename.c_str(), id, imageSize, imageProvider, currentFps, currentCodec, currentMaxSegmentFrames, tmpBuffer, stdPass));
    }
    catch (const std::exception& e)
    {
        printMsg(kit.msgLog, STR("OutImgAvi: STL exception: %0"), e.what(), msgErr);
        returnFalse;
    }

    returnTrue;
}

//================================================================
//
// OutImgAviImpl::saveImage
//
//================================================================

template <typename Element>
stdbool OutImgAviImpl::saveImage(const Matrix<const Element>& image, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    ImageProviderMemcpy<Element> imageProvider(image, kit);
    return saveImageGeneric(image.size(), imageProvider, desc, id, stdPassThru);
}

//================================================================
//
// OutImgAviImpl::saveImage
//
//================================================================

stdbool OutImgAviImpl::saveImage(const Point<Space>& imageSize, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit))
{
    return saveImageGeneric(imageSize, imageProvider, desc, id, stdPassThru);
}

//----------------------------------------------------------------

}

