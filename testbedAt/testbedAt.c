#include "at_client.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <new>

#include "errorLog/errorLog.h"
#include "numbers/int/intType.h"
#include "userOutput/msgLog.h"
#include "userOutput/errorLogEx.h"
#include "atInterface/atInterface.h"
#include "checkHeap.h"
#include "formattedOutput/userOutputThunks.h"
#include "formattedOutput/messageFormatterStdio.h"
#include "atAssembly/atAssembly.h"
#include "dataAlloc/arrayMemory.h"
#include "threading/threadManagerImpl.h"
#include "errorLog/debugBreak.h"
#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "allocation/mallocFlatAllocator/mallocFlatAllocator.h"
#include "compileTools/classContext.h"
#include "kits/setBusyStatus.h"
#include "baseInterfaces/baseSignals.h"

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
    REQUIRE_AT_EX(condition, false, return)

#define REQUIRE_AT_EX(condition, finalExit, returnStatement) \
    \
    if (condition) \
        ; \
    else \
    { \
        reportError(api, CT(__FILE__) CT("(") \
            CT(PREP_STRINGIZE(__LINE__)) CT("): ") \
            CT(PREP_STRINGIZE(condition)) CT(" failed"), finalExit); \
        returnStatement; \
    }

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// OutputLogByAt
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// OutputLogByAt
//
//================================================================

template <typename AtApi>
class OutputLogByAt : public MsgLog
{

public:

    using PrintFunc = at_bool AT_CALL (const AtApi* api, const at_char* message, at_msg_kind msg_kind);
    using ClearFunc = at_bool AT_CALL (const AtApi* api);
    using UpdateFunc = at_bool AT_CALL (const AtApi* api);

    struct OutputFuncs
    {
        PrintFunc* print;
        ClearFunc* clear;
        UpdateFunc* update;
    };

public:

    OutputLogByAt
    (
        const OutputFuncs& func,
        const OutputFuncs& aux,
        const AtApi* api,
        bool useDebugOutput,
        MessageFormatter& formatter
    )
        :
        func(func),
        aux(aux),
        api(api),
        useDebugOutput(useDebugOutput),
        formatter(formatter)
    {
    }

    bool isThreadProtected() const
    {
        return false;
    }

    void lock()
    {
    }

    void unlock()
    {
    }

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
    {
        ensure(func.clear(api) != 0);

        if (aux.clear)
            ensure(aux.clear(api) != 0);

        return true;
    }

    bool update()
    {
        ensure(func.update(api) != 0);

        if (aux.update)
            ensure(aux.update(api) != 0);

        return true;
    }

private:

    const AtApi* const api;

    OutputFuncs const func;
    OutputFuncs const aux;

    bool const useDebugOutput;

    MessageFormatter& formatter;

};

//================================================================
//
// OutputLogByAt::addMsg
//
//================================================================

template <typename AtApi>
bool OutputLogByAt<AtApi>::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    formatter.clear();
    v.func(v.value, formatter);
    ensure(formatter.valid());

    ensure(func.print(api, formatter.data(), at_msg_kind(msgKind)) != 0);

    if (aux.print)
        ensure(aux.print(api, formatter.data(), at_msg_kind(msgKind)) != 0);

#if defined(_WIN32)

    if (useDebugOutput)
    {
        formatter.write(CT("\n"), 1);
        ensure(formatter.valid());
        OutputDebugString(formatter.data());
    }

#endif

    return true;
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// SetBusyStatusByAt
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <typename AtApi>
class SetBusyStatusByAt : public SetBusyStatus
{

public:

    SetBusyStatusByAt(const AtApi* api, MessageFormatter& formatter)
        : api(api), formatter(formatter) {}

    bool set(const FormatOutputAtom& message);

    bool reset()
    {
        return api->set_busy_status(api, CT("Processing in client module")) != 0;
    }

private:

    const AtApi* const api;
    MessageFormatter& formatter;

};

//================================================================
//
// SetBusyStatusByAt<AtApi>::operator () 
//
//================================================================

template <typename AtApi>
bool SetBusyStatusByAt<AtApi>::set(const FormatOutputAtom& message)
{
    formatter.clear();
    message.func(message.value, formatter);
    ensure(formatter.valid());

    ensure(api->set_busy_status(api, formatter.data()) != 0);

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
class AtImgConsoleImplThunk : public BaseImageConsole
{

public:

    stdbool addImage(const Matrix<const uint8>& img, const ImgOutputHint& hint, bool dataProcessing, stdNullPars)
    {
        if_not (dataProcessing)
            return;

        ////

        formatter.clear();
        hint.desc.func(hint.desc.value, formatter);
        require(formatter.valid());

        ////

        MATRIX_EXPOSE(img);

        require
        (
            api->outimg_gray8(api, unsafePtr(imgMemPtr, imgSizeX, imgSizeY),
                imgMemPitch, imgSizeX, imgSizeY,
                hint.id, hint.minSize.X, hint.minSize.Y, hint.newLine,
                formatter.data()) != 0
        );

        returnTrue;
    }

    stdbool addImage(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, bool dataProcessing, stdNullPars)
    {
        if_not (dataProcessing)
            returnTrue;

        ////

        formatter.clear();
        hint.desc.func(hint.desc.value, formatter);
        require(formatter.valid());

        ////

        MATRIX_EXPOSE(img);

        const uint8_x4* imgPtr = unsafePtr(imgMemPtr, imgSizeX, imgSizeY);
        COMPILE_ASSERT_EQUAL_LAYOUT(at_pixel_rgb32, uint8_x4);

        require
        (
            api->outimg_rgb32
            (
                api, 
                (const at_pixel_rgb32*) imgPtr, imgMemPitch, imgSizeX, imgSizeY,
                hint.id, hint.minSize.X, hint.minSize.Y, hint.newLine,
                formatter.data()
            ) 
            != 0
        );

        returnTrue;
    }

    stdbool clear(stdNullPars)
    {
        require(api->outimg_clear(api) != 0);
        returnTrue;
    }

    stdbool update(stdNullPars)
    {
        require(api->outimg_update(api) != 0);
        returnTrue;
    }

public:

    inline AtImgConsoleImplThunk(const AtApi* api, MessageFormatter& formatter)
        : api(api), formatter(formatter) {}

private:

    const AtApi* const api;
    MessageFormatter& formatter;

};

//================================================================
//
// Check actions IDs.
//
//================================================================

COMPILE_ASSERT(baseActionId::MouseLeftDown == AT_ACTION_ID_MOUSE_LEFT_DOWN);
COMPILE_ASSERT(baseActionId::MouseLeftUp == AT_ACTION_ID_MOUSE_LEFT_UP);

COMPILE_ASSERT(baseActionId::MouseRightDown == AT_ACTION_ID_MOUSE_RIGHT_DOWN);
COMPILE_ASSERT(baseActionId::MouseRightUp == AT_ACTION_ID_MOUSE_RIGHT_UP);

COMPILE_ASSERT(baseActionId::WheelDown == AT_ACTION_ID_WHEEL_DOWN);
COMPILE_ASSERT(baseActionId::WheelUp == AT_ACTION_ID_WHEEL_UP);

//================================================================
//
// AtSignalSetThunk
//
//================================================================

template <typename AtApi>
class AtSignalSetThunk : public BaseActionSetup
{

public:

    bool actsetClear()
        {return api->actset_clear(api) != 0;}

    bool actsetUpdate()
        {return api->actset_update(api) != 0;}

    bool actsetAdd(BaseActionId id, const CharType* key, const CharType* name, const CharType* comment)
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

class AtSignalTestThunk : public BaseActionReceiving
{

public:

    virtual void getActions(BaseActionReceiver& receiver)
    {
        auto count = api->action_count(api);

        for_count (i, count)
        {
            BaseActionId id{};

            if_not (api->action_item(api, i, &id) != 0)
                continue;

            receiver.process(makeArray(&id, 1));
        }
    }

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

    AtImageProviderThunk(BaseImageProvider& imageProvider, const TraceScope& trace)
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
        const TraceScope& trace = self->trace;

        COMPILE_ASSERT_EQUAL_LAYOUT(at_pixel_rgb32, uint8_x4);
        return errorBlock(self->imageProvider.saveBgr32(Matrix<uint8_x4>((uint8_x4*) mem_ptr, mem_pitch, size_X, size_Y), stdNullPass));
    }

private:

    TraceScope trace;
    BaseImageProvider& imageProvider;

};

//================================================================
//
// AtVideoOverlayThunk
//
//================================================================

class AtVideoOverlayThunk : public BaseVideoOverlay
{

public:

    using Kit = LocalLogKit;

public:

    stdbool setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        AtImageProviderThunk atProvider(imageProvider, trace);

        if_not (dataProcessing)
        {
            // Imitates the processing on counting phase. The pitch on execution phase may differ, 
            // but the provider implementation is tolerant to it up to some maximal row alignment.

            require(atProvider.callbackFunc(&atProvider, nullptr, imageProvider.desiredPitch(), size.X, size.Y) != 0);
        }
        else
        {
            if (textEnabled) 
                printMsg(kit.localLog, STR("OVERLAY: %"), desc);

            require(api->video_image_set(api, size.X, size.Y, &atProvider, atProvider.callbackFunc) != 0);
        }

        returnTrue;
    }

    stdbool setImageFake(stdNullPars)
    {
        returnTrue;
    }

    stdbool updateImage(stdNullPars)
    {
        at_bool result = api->video_image_update(api);
        require(DEBUG_BREAK_CHECK(result != 0));
        returnTrue;
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

    virtual stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, stdNullPars)
    {
        CRITSEC_GUARD(lock);
        AtImageProviderThunk atProvider(imageProvider, trace);

        require(base.set_image(base.context, size.X, size.Y, &atProvider, atProvider.callbackFunc) != 0);
        returnTrue;
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

    stdbool init(stdPars(InitKit))
    {
        require(kit.threadManager.createCriticalSection(lock, stdPass));
        returnTrue;
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
    MessageFormatterStdio formatter{makeArray(client->formatterArray, client->formatterSize)}; \
    \
    OutputLogByAt<AtApi> msgLog({api->gcon_print_ex, api->gcon_clear, api->gcon_update}, \
        {api->lcon_print_ex, api->lcon_clear, api->lcon_update}, api, true, formatter); \
    \
    OutputLogByAt<AtApi> localLog({api->lcon_print_ex, api->lcon_clear, api->lcon_update}, \
        {nullptr, nullptr, nullptr}, api, false, formatter); \
    \
    ErrorLogByMsgLog errorLog(msgLog); \
    ErrorLogKit errorLogKit(errorLog); \
    ErrorLogExByMsgLog errorLogEx(msgLog); \
    \
    MAKE_MALLOC_ALLOCATOR_OBJECT(errorLogKit); \
    \
    AtImgConsoleImplThunk<AtApi> imgConsole(api, formatter); \
    \
    AtSignalSetThunk<AtApi> signalSet(api); \
    ThreadManagerImpl threadManager; \
    \
    SetBusyStatusByAt<AtApi> setBusyStatus(api, formatter); \
    \
    auto baseKit = kitCombine \
    ( \
        MessageFormatterKit(formatter), \
        ErrorLogKit(errorLog), \
        ErrorLogExKit(errorLogEx), \
        MsgLogKit(msgLog), \
        LocalLogKit(localLog), \
        LocalLogAuxKit(false, localLog), \
        AtImgConsoleKit(imgConsole), \
        AtSignalSetKit(signalSet), \
        MallocKit(mallocAllocator), \
        ThreadManagerKit(threadManager), \
        SetBusyStatusKit(setBusyStatus) \
    );

//================================================================
//
// getVideoName
//
//================================================================

stdbool getVideoName(const at_api_process& api, ArrayMemory<CharType>& result, AllocatorObject<CpuAddrU>& allocator, stdPars(ErrorLogKit))
{
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

    returnTrue;
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
    ensure(api.videofile_pos(&api, &atIndex, &atCount) != 0);

    ensure(0 <= atIndex && atIndex <= spaceMax);
    index = atIndex;

    ensure(atCount >= 0 && atCount <= spaceMax);
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

    static constexpr int formatterSize = 65536;
    CharType formatterArray[formatterSize];
};

//================================================================
//
// atClientCreate
//
//================================================================

stdbool atClientCreateCore(void** instance, const at_api_create* api, const AtEngineFactory& engineFactory)
{
    *instance = 0;

    ////

    Client* client = new (std::nothrow) Client;
    REMEMBER_CLEANUP1_EX(clientCleanup, delete client, Client*, client);

    if_not (client)
    {
        api->gcon_print_ex(api, CT("Cannot allocate AT client instance"), at_msg_err);
        returnFalse;
    }

    ////

    stdTraceRoot;
    COMMON_KIT_IMPL(at_api_create, kit);

    ////

    require(client->asyncOverlay.init(stdPass));

    ////

    at_async_overlay async_overlay;
    REQUIRE(api->get_async_overlay(api, &async_overlay) != 0);
    client->asyncOverlay.setBase(async_overlay);

    ////

    require(client->assembly.init(engineFactory, stdPass));

    ////

    clientCleanup.cancel();
    *instance = client;

    returnTrue;
}

//----------------------------------------------------------------

void atClientCreate(void** instance, const at_api_create* api, const AtEngineFactory& engineFactory)
{
    errorBlock(atClientCreateCore(instance, api, engineFactory));
}

//================================================================
//
// atClientDestroy
//
//================================================================

void atClientDestroy(void* instance, const at_api_destroy* api)
{
    bool finalExit = api->final_exit(api) != 0;
    
    Client* client = (Client*) instance;
    REQUIRE_AT_EX(client != 0, finalExit, return);

    ////

    stdTraceRoot;
    COMMON_KIT_IMPL(at_api_destroy, kit);

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

    stdbool alloc(AllocatorState& state, AddrU size, SpaceU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
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

stdbool atClientProcessCore(void* instance, const at_api_process* api)
{
    Client* client = static_cast<Client*>(instance);
    REQUIRE_AT_EX(client != 0, false, returnFalse);

    ////

    stdTraceRoot;

    COMMON_KIT_IMPL(at_api_process, baseKit);

    //----------------------------------------------------------------
    //
    // Monitor mallocs
    //
    //----------------------------------------------------------------

    MallocMonitorThunk<CpuAddrU> mallocMonitor(baseKit.malloc.func);
    AllocatorObject<CpuAddrU> mallocMonitorObject(baseKit.malloc.state, mallocMonitor);

    auto kit = kitReplace(baseKit, MallocKit(mallocMonitorObject));

    REMEMBER_CLEANUP
    (
        if (mallocMonitor.counter)
        {
            // printMsg(kit.localLog, STR("In-Process Malloc Count: %0"), mallocMonitor.counter, msgWarn);
        }
    );

    //----------------------------------------------------------------
    //
    // Video frame
    //
    //----------------------------------------------------------------

    const at_pixel_rgb32* frameMemPtrAt = 0;
    Space frameMemPitch = 0;
    Space frameSizeX = 0;
    Space frameSizeY = 0;
    REQUIRE(api->video_image_get(api, &frameMemPtrAt, &frameMemPitch, &frameSizeX, &frameSizeY) != 0);

    COMPILE_ASSERT_EQUAL_LAYOUT(at_pixel_rgb32, uint8_x4);
    const uint8_x4* frameMemPtr = (const uint8_x4*) frameMemPtrAt;

    if (frameSizeX == 0 || frameSizeY == 0)
    {
        frameMemPtr = 0;
        frameSizeX = 0;
        frameSizeY = 0;
        frameMemPitch = 0;
    }

    Matrix<const uint8_x4> frame;
    REQUIRE(frame.assign(frameMemPtr, frameMemPitch, frameSizeX, frameSizeY) != 0);

    //----------------------------------------------------------------
    //
    // Video info
    //
    //----------------------------------------------------------------

    if_not (errorBlock(getVideoName(*api, client->videofileName, mallocAllocator, stdPass)))
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

    ARRAY_EXPOSE_UNSAFE_EX(client->videofileName, videofileName);

    AtVideoInfo atVideoInfo
    (
        charArrayFromPtr(videofileNamePtr),
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
    REQUIRE(api->get_continuous_mode(api, &atRunning, &atPlaying) != 0);

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    AtVideoFrame atVideoFrame(frame);
    AtVideoOverlayThunk atVideoOverlay(api, kit);
    AtSignalTestThunk atSignalTest(api);

    auto processKit = kitCombine
    (
        kit,
        AtVideoFrameKit(atVideoFrame),
        AtVideoOverlayKit(atVideoOverlay),
        AtAsyncOverlayKit(client->asyncOverlay),
        AtSignalTestKit(atSignalTest),
        AtVideoInfoKit(atVideoInfo),
        AtUserPointKit(userPointValid, userPoint),
        AtContinousModeKit(atRunning != 0, atPlaying != 0)
    );

    ////

    require(client->assembly.process(stdPassKit(processKit)));

    returnTrue;
}

//----------------------------------------------------------------

void atClientProcess(void* instance, const at_api_process* api)
{
    errorBlock(atClientProcessCore(instance, api));
}
