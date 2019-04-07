#include "at_client.h"

#include <windows.h>
#include <new>
#include <sstream>

#include "errorLog/errorLog.h"
#include "numbers/int/intType.h"
#include "userOutput/msgLog.h"
#include "userOutput/errorLogEx.h"
#include "atInterface/atInterface.h"
#include "checkHeap.h"
#include "formattedOutput/userOutputThunks.h"
#include "formattedOutput/formatStreamStl.h"
#include "atAssembly/atAssembly.h"
#include "dataAlloc/arrayMemory.h"
#include "threading/threadManagerImpl.h"
#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "allocation/mallocFlatAllocator/mallocFlatAllocator.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Base AT layer:
//
// * C++ heap check.
// * Error reporting.
// * Output logs.
// * AT video frame.
// * AT image console.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// reportError
//
//================================================================

template <typename PrintApi>
void reportError(const PrintApi* api, const CharType* message, bool finalExit = false)
{
    api->gcon_print_ex(api, message, at_msg_err);

    if (!finalExit)
        MessageBeep(MB_ICONHAND);
    else
        MessageBox(NULL, message, CT("Algorithm Module Error"), MB_ICONHAND);
}

//================================================================
//
// REQUIRE_AT
//
//================================================================

#define REQUIRE_AT(condition) \
    REQUIRE_AT_EX(condition, false)

#define REQUIRE_AT_EX(condition, finalExit) \
    \
    if (condition) \
        ; \
    else \
    { \
        reportError(api, CT(__FILE__) CT("(") \
            CT(PREP_STRINGIZE(__LINE__)) CT("): ") \
            CT(PREP_STRINGIZE(condition)) CT(" failed"), finalExit); \
        return; \
    }

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// OutputLogToAt
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// OutputLogToAt
//
//================================================================

template <typename AtApi>
class OutputLogToAt : public MsgLog
{

public:

    using PrintFunc = at_bool AT_CALL (const AtApi* api, const at_char* message, at_msg_kind msg_kind);
    using ClearFunc = at_bool AT_CALL (const AtApi* api);
    using UpdateFunc = at_bool AT_CALL (const AtApi* api);

public:

    OutputLogToAt
    (
        PrintFunc* printFunc, ClearFunc* clearFunc, UpdateFunc* updateFunc,
        const AtApi* api,
        bool useDebugOutput
    )
        :
        printFunc(printFunc), clearFunc(clearFunc), updateFunc(updateFunc),
        api(api), useDebugOutput(useDebugOutput)
    {
    }

    bool isThreadProtected() const
        {return false;}

    void lock()
        {}

    void unlock()
        {}

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
        {return clearFunc(api) != 0;}

    bool update()
        {return updateFunc(api) != 0;}

private:

    PrintFunc* const printFunc;
    ClearFunc* const clearFunc;
    UpdateFunc* const updateFunc;

    const AtApi* const api;

    bool const useDebugOutput;

};

//================================================================
//
// OutputLogToAt::addMsg
//
//================================================================

template <typename AtApi>
bool OutputLogToAt<AtApi>::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    try
    {
        using namespace std;

        std::basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        v.func(v.value, formatToStream);
        require(formatToStream.isOk());
        require(!!stringStream);

        require(printFunc(api, stringStream.rdbuf()->str().c_str(), at_msg_kind(msgKind)) != 0);

    #if defined(_WIN32)
        stringStream << endl;

        if (useDebugOutput)
            OutputDebugString(stringStream.rdbuf()->str().c_str());
    #endif

    }
    catch (const std::exception&) {require(false);}

    return true;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// AtImgConsoleImplThunk
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <typename AtApi>
class AtImgConsoleImplThunk : public AtImgConsole
{

public:

    bool addImageFunc(const Matrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
    {
        std::basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        hint.desc.func(hint.desc.value, formatToStream);
        require(formatToStream.isOk());
        require(!!stringStream);

        ////

        MATRIX_EXPOSE(img);

        require
        (
            api->outimg_gray8(api, unsafePtr(imgMemPtr, imgSizeX, imgSizeY),
                imgMemPitch, imgSizeX, imgSizeY,
                hint.id, hint.minSize.X, hint.minSize.Y, hint.newLine,
                stringStream.rdbuf()->str().c_str()) != 0
        );

        return true;
    }

    bool addImageFunc(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
    {
        MATRIX_EXPOSE(img);

        ////

        std::basic_stringstream<CharType> stringStream;
        FormatStreamStlThunk formatToStream(stringStream);

        hint.desc.func(hint.desc.value, formatToStream);
        require(formatToStream.isOk());
        require(!!stringStream);
        const CharType* textDesc = stringStream.rdbuf()->str().c_str();

        ////

        const uint8_x4* imgPtr = unsafePtr(imgMemPtr, imgSizeX, imgSizeY);
        COMPILE_ASSERT(sizeof(at_pixel_rgb32) == sizeof(uint8_x4));

        require
        (
            api->outimg_rgb32(api, (const at_pixel_rgb32*) imgPtr,
                imgMemPitch, imgSizeX, imgSizeY,
                hint.id, hint.minSize.X, hint.minSize.Y, hint.newLine,
                stringStream.rdbuf()->str().c_str()) != 0
        );

        return true;
    }

    bool clear(stdNullPars)
    {
        require(api->outimg_clear(api) != 0);
        return true;
    }

    bool update(stdNullPars)
    {
        require(api->outimg_update(api) != 0);
        return true;
    }

public:

    inline AtImgConsoleImplThunk(const AtApi* api)
        : api(api) {}

private:

    const AtApi* const api;

};

//================================================================
//
// AtSignalSetThunk
//
//================================================================

template <typename AtApi>
class AtSignalSetThunk : public AtSignalSet
{

public:

    bool actsetClear()
        {return api->actset_clear(api) != 0;}

    bool actsetUpdate()
        {return api->actset_update(api) != 0;}

    bool actsetAdd(AtActionId id, const CharType* name, const CharType* key, const CharType* comment)
        {return api->actset_add(api, id, key, name, comment) != 0;}

public:

    inline AtSignalSetThunk(const AtApi* api)
        : api(api) {}

private:

    const AtApi* const api;

};

//================================================================
//
// AtSignalTestThunk
//
//================================================================

class AtSignalTestThunk : public AtSignalTest
{

public:

    int32 actionCount()
        {return api->action_count(api);}

    bool actionItem(int32 index, AtActionId& id)
        {return api->action_item(api, index, &id) != 0;}

public:

    inline AtSignalTestThunk(const at_api_process* api)
        : api(api) {}

private:

    const at_api_process* const api;

};

//================================================================
//
// AtImageProviderThunk
//
//================================================================

class AtImageProviderThunk
{

public:

    AtImageProviderThunk(AtImageProvider<uint8_x4>& imageProvider, const TraceScope& trace)
        : imageProvider(imageProvider), trace(trace) {}

    static at_bool AT_CALL callbackFunc
    (
        void* context,
        at_pixel_rgb32* mem_ptr,
        at_image_space mem_pitch,
        at_image_space size_X,
        at_image_space size_Y
    )
    {
        AtImageProviderThunk* self = (AtImageProviderThunk*) context;
        const TraceScope& TRACE_SCOPE(stdTraceName) = self->trace;

        COMPILE_ASSERT(sizeof(at_pixel_rgb32) == sizeof(uint8_x4));
        return self->imageProvider.saveImage(Matrix<uint8_x4>((uint8_x4*) mem_ptr, mem_pitch, size_X, size_Y), stdNullPass);
    }

private:

    TraceScope trace;
    AtImageProvider<uint8_x4>& imageProvider;

};

//================================================================
//
// AtVideoOverlayThunk
//
//================================================================

class AtVideoOverlayThunk : public AtVideoOverlay
{

public:

    using Kit = LocalLogKit;

public:

    bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        TRACE_REASSEMBLE(stdTraceName);
        if (textEnabled) printMsg(kit.localLog, STR("OVERLAY: %0"), desc);
        AtImageProviderThunk atProvider(imageProvider, TRACE_SCOPE(stdTraceName));
        return api->video_image_set(api, size.X, size.Y, &atProvider, atProvider.callbackFunc) != 0;
    }

    bool setFakeImage(stdNullPars)
    {
        return true;
    }

    bool updateImage(stdNullPars)
    {
        at_bool result = api->video_image_update(api);
        DEBUG_BREAK_CHECK(result != 0);
        return result != 0;
    }

public:

    inline AtVideoOverlayThunk(const at_api_process* api, const Kit& kit)
        : api(api), kit(kit) {}

private:

    const at_api_process* const api;
    const Kit kit;

};

//================================================================
//
// AtAsyncOverlayImpl
//
//================================================================

class AtAsyncOverlayImpl : public AtAsyncOverlay
{

public:

    virtual bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, stdNullPars)
    {
        CRITSEC_GUARD(lock);
        TRACE_REASSEMBLE(stdTraceName);
        AtImageProviderThunk atProvider(imageProvider, TRACE_SCOPE(stdTraceName));

        return base.set_image(base.context, size.X, size.Y, &atProvider, atProvider.callbackFunc) != 0;
    }

    AtAsyncOverlayImpl()
    {
        base.context = 0;
        base.set_image = 0;
    }

    ~AtAsyncOverlayImpl()
    {
    }

public:

    void setBase(const at_async_overlay& base)
    {
        this->base = base;
    }

public:

    KIT_COMBINE2(InitKit, ThreadManagerKit, ThreadToolKit);

    bool init(stdPars(InitKit))
    {
        stdBegin;
        require(kit.threadManager.createCriticalSection(lock, stdPass));
        stdEnd;
    }

private:

    CriticalSection lock;
    at_async_overlay base;

};

//================================================================
//
// COMMON_KIT_IMPL
//
//================================================================

#define COMMON_KIT_IMPL(AtApi, baseKit) \
    \
    OutputLogToAt<AtApi> msgLog(api->gcon_print_ex, api->gcon_clear, api->gcon_update, api, true); \
    OutputLogToAt<AtApi> localLog(api->lcon_print_ex, api->lcon_clear, api->lcon_update, api, false); \
    \
    ErrorLogThunk errorLog(msgLog); \
    ErrorLogKit errorLogKit(errorLog, 0); \
    ErrorLogExThunk errorLogEx(msgLog); \
    \
    MAKE_MALLOC_ALLOCATOR_OBJECT(errorLogKit); \
    \
    AtImgConsoleImplThunk<AtApi> imgConsole(api); \
    \
    AtSignalSetThunk<AtApi> signalSet(api); \
    ThreadManagerImpl threadManager; \
    \
    atStartup::InitKit baseKit = kitCombine \
    ( \
        ErrorLogKit(errorLog, 0), \
        ErrorLogExKit(errorLogEx, 0), \
        MsgLogKit(msgLog, 0), \
        LocalLogKit(localLog, 0), \
        LocalLogAuxKit(false, localLog), \
        AtImgConsoleKit(imgConsole, 0), \
        AtSignalSetKit(signalSet, 0), \
        MallocKit(mallocAllocator, 0), \
        ThreadManagerKit(threadManager, 0) \
    );

//================================================================
//
// getVideoName
//
//================================================================

bool getVideoName(const at_api_process& api, ArrayMemory<CharType>& result, AllocatorObject<CpuAddrU>& allocator, stdPars(ErrorLogKit))
{
    stdBegin;

    size_t atSizeApi = 0;
    require(api.videofile_name(&api, NULL, 0, &atSizeApi) != 0);
    Space atSize = 0;
    REQUIRE(convertExact(atSizeApi, atSize));

    ////

    if_not (result.resize(atSize))
        require(result.realloc(atSize, cpuBaseByteAlignment, allocator, stdPass));

    ////

    ARRAY_EXPOSE(result);

    size_t atFullSize = 0;
    require(api.videofile_name(&api, unsafePtr(resultPtr, resultSize), resultSize, &atFullSize) != 0);
    require(atFullSize <= spaceMax);

    ////

    require(resultSize == atFullSize);

    stdEnd;
}

//================================================================
//
// getVideoPosition
//
//================================================================

bool getVideoPosition(const at_api_process& api, Space& index, Space& count)
{
    uint32 atIndex = 0;
    uint32 atCount = 0;
    require(api.videofile_pos(&api, &atIndex, &atCount) != 0);

    require(0 <= atIndex && atIndex <= spaceMax);
    index = atIndex;

    require(atCount >= 0 && atCount <= spaceMax);
    count = atCount;

    return true;
}

//================================================================
//
// getVideoInterlaced
//
//================================================================

template <typename Bool>
void getVideoInterlaced(const at_api_process& api, Bool& interlacedMode, Bool& interlacedLower)
{
    at_bool atInterlacedMode = 0;
    at_bool atInterlacedLower = 0;

    api.get_interlaced(&api, &atInterlacedMode, &atInterlacedLower);

    interlacedMode = (atInterlacedMode != 0);
    interlacedLower = (atInterlacedLower != 0);
}

//================================================================
//
// Client
//
//================================================================

struct Client
{
    AtAsyncOverlayImpl asyncOverlay; // should be declared before the assembly!

    atStartup::AtAssembly assembly;
    ArrayMemory<CharType> videofileName;
};

//================================================================
//
// atClientCreate
//
//================================================================

void atClientCreate(void** instance, const at_api_create* api, const AtEngineFactory& engineFactory)
{
    *instance = 0;

    ////

    TRACE_ROOT_STD;
    COMMON_KIT_IMPL(at_api_create, kit);

    ////

    Client* client = new (std::nothrow) Client;
    REQUIREV(client != 0);
    REMEMBER_CLEANUP1_EX(clientCleanup, delete client, Client*, client);

    ////

    requirev(client->asyncOverlay.init(stdPass));

    ////

    at_async_overlay async_overlay;
    REQUIREV(api->get_async_overlay(api, &async_overlay) != 0);
    client->asyncOverlay.setBase(async_overlay);

    ////

    requirev(client->assembly.init(engineFactory, stdPass));

    ////

    clientCleanup.cancel();
    *instance = client;
}

//================================================================
//
// atClientDestroy
//
//================================================================

void atClientDestroy(void* instance, const at_api_destroy* api)
{
    TRACE_ROOT_STD;
    COMMON_KIT_IMPL(at_api_destroy, kit);

    ////

    bool finalExit = api->final_exit(api) != 0;

    ////

    Client* client = (Client*) instance;
    REQUIRE_AT_EX(client != 0, finalExit);

    ////

    client->assembly.finalize(stdPass);
    delete client;

    ////

    api->lcon_clear(api);
    api->outimg_clear(api);

    ////

    if (!checkHeapIntegrity())
        reportError(api, CT("Heap memory is damaged!"), finalExit);
    else if (!checkHeapLeaks())
        reportError(api, CT("Memory leaks are detected!"), finalExit);
}

//================================================================
//
// MallocMonitorThunk
//
//================================================================

template <typename AddrU>
class MallocMonitorThunk : public AllocatorInterface<AddrU>
{

public:

    inline MallocMonitorThunk(AllocatorInterface<AddrU>& base)
        : base(base) {}

    bool alloc(AllocatorState& state, AddrU size, SpaceU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
    {
        ++counter;
        return base.alloc(state, size, alignment, owner, result, stdNullPassThru);
    }

public:

    int32 counter = 0;

private:

    AllocatorInterface<AddrU>& base;

};

//================================================================
//
// atClientProcess
//
//================================================================

void atClientProcess(void* instance, const at_api_process* api)
{
    TRACE_ROOT_STD;

    COMMON_KIT_IMPL(at_api_process, baseKit);

    //----------------------------------------------------------------
    //
    // Monitor mallocs
    //
    //----------------------------------------------------------------

    MallocMonitorThunk<CpuAddrU> mallocMonitor(baseKit.malloc.func);
    AllocatorObject<CpuAddrU> mallocMonitorObject(baseKit.malloc.state, mallocMonitor);

    atStartup::InitKit kit = kitReplace(baseKit, MallocKit(mallocMonitorObject, 0));

    //----------------------------------------------------------------
    //
    // Video frame
    //
    //----------------------------------------------------------------

    const at_pixel_rgb32* frameMemPtrAt = 0;
    Space frameMemPitch = 0;
    Space frameSizeX = 0;
    Space frameSizeY = 0;
    REQUIREV(api->video_image_get(api, &frameMemPtrAt, &frameMemPitch, &frameSizeX, &frameSizeY) != 0);

    COMPILE_ASSERT(sizeof(at_pixel_rgb32) == sizeof(uint8_x4));
    const uint8_x4* frameMemPtr = (const uint8_x4*) frameMemPtrAt;

    if (frameSizeX == 0 || frameSizeY == 0)
    {
        frameMemPtr = 0;
        frameSizeX = 0;
        frameSizeY = 0;
        frameMemPitch = 0;
    }

    Matrix<const uint8_x4> frame;
    REQUIREV(frame.assign(frameMemPtr, frameMemPitch, frameSizeX, frameSizeY) != 0);

    //----------------------------------------------------------------
    //
    // Video info
    //
    //----------------------------------------------------------------

    Client* client = static_cast<Client*>(instance);
    REQUIREV(client != 0);

    ////

    if_not (getVideoName(*api, client->videofileName, mallocAllocator, stdPass))
        client->videofileName.resizeNull();

    ////

    Space frameIndex = 0;
    Space frameCount = 1;

    if_not (getVideoPosition(*api, frameIndex, frameCount))
        {frameIndex = 0; frameCount = 1;}

    ////

    bool interlacedMode = false;
    bool interlacedLower = false;
    getVideoInterlaced(*api, interlacedMode, interlacedLower);

    ////

    ARRAY_EXPOSE_UNSAFE(client->videofileName, videofileName);

    AtVideoInfo atVideoInfo
    (
        CharArray(videofileNamePtr, videofileNameSize),
        frameIndex, frameCount,
        interlacedMode, interlacedLower,
        frame.size()
    );

    ////

    Point<Space> userPoint = point(0);
    bool userPointValid = api->get_cursor(api, &userPoint.X, &userPoint.Y) != 0;

    ////

    at_bool atRunning = false;
    at_bool atPlaying = false;
    requirev(api->get_continuous_mode(api, &atRunning, &atPlaying) != 0);

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    AtVideoFrame atVideoFrame(frame);
    AtVideoOverlayThunk atVideoOverlay(api, kit);
    AtSignalTestThunk atSignalTest(api);

    atStartup::ProcessKit processKit = kitCombine
    (
        kit,
        AtVideoFrameKit(atVideoFrame, 0),
        AtVideoOverlayKit(atVideoOverlay, 0),
        AtAsyncOverlayKit(client->asyncOverlay, 0),
        AtSignalTestKit(atSignalTest, 0),
        AtVideoInfoKit(atVideoInfo, 0),
        AtUserPointKit(userPointValid, userPoint),
        AtContinousModeKit(atRunning != 0, atPlaying != 0)
    );

    ////

    client->assembly.process(stdPassKit(processKit));

    //----------------------------------------------------------------
    //
    // Monitor malloc usage
    //
    //----------------------------------------------------------------

    if (mallocMonitor.counter)
        printMsg(kit.localLog, STR("In-Process Malloc Count: %0"), mallocMonitor.counter, msgWarn);

}
