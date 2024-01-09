#include "overlayBuffer.h"

#include "gpuBuffer/gpuBuffer.h"
#include "gpuProcessKit.h"
#include "storage/rememberCleanup.h"
#include "lib/imageTools/getAlignedBufferPitch.h"

namespace overlayBuffer {

//================================================================
//
// OverlayHolder
//
// Greedy image buffer to use for the main BGRx picture.
//
//================================================================

class OverlayHolder
{

public:

    void dealloc()
    {
        eventsDealloc();
        buffer.dealloc();
        resetImage();
    }

    ////

    void resetImage()
    {
        imageSet = false;
        image = {};
    }

    bool hasImage() const {return imageSet;}

    auto getImage() const {return image;}

    ////

    template <typename Kit>
    stdbool setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(Kit));

    ////

    template <typename Kit>
    stdbool useImage(ImageUser& imageUser, stdPars(Kit));

    //----------------------------------------------------------------
    //
    // Exchange.
    //
    //----------------------------------------------------------------

    friend inline void exchange(OverlayHolder& a, OverlayHolder& b)
    {
        ::exchange(a.eventsAllocated, b.eventsAllocated);
        exchange(a.readingCompletion, b.readingCompletion);
        exchange(a.writingCompletion, b.writingCompletion);

        exchange(a.buffer, b.buffer);

        ::exchange(a.imageSet, b.imageSet);
        exchangeByCopying(a.image, b.image);
    }

    //----------------------------------------------------------------
    //
    // Events allocation.
    //
    //----------------------------------------------------------------

private:

    void eventsDealloc()
    {
        eventsAllocated = false;
        readingCompletion.clear();
        writingCompletion.clear();
   }

    ////

    template <typename Kit>
    stdbool eventsEnsureAlloc(stdPars(Kit))
    {
        if (eventsAllocated)
            returnTrue;

        ////

        REMEMBER_CLEANUP_EX(errorCleanup, eventsDealloc());

        require(kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, readingCompletion, stdPass));
        require(kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, writingCompletion, stdPass));

        ////

        errorCleanup.cancel();
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    bool eventsAllocated = false;
    GpuEventOwner readingCompletion;
    GpuEventOwner writingCompletion;

    GpuBuffer buffer;

    bool imageSet = false;
    GpuMatrix<Type> image;

};

//================================================================
//
// OverlayHolder::useImage
//
//================================================================

template <typename Kit>
stdbool OverlayHolder::useImage(ImageUser& imageUser, stdPars(Kit))
{

    //----------------------------------------------------------------
    //
    // If there is no image, use a simple path without data access.
    //
    //----------------------------------------------------------------

    if_not (imageSet)
    {
        require(imageUser(false, {}, stdPass));
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Before using the buffer insert a barrier into our stream
    // to wait for the completion of the other stream buffer usage.
    //
    //----------------------------------------------------------------

    if (kit.dataProcessing)
        require(kit.gpuEventRecording.putEventDependency(writingCompletion, kit.gpuCurrentStream, stdPass));

    //----------------------------------------------------------------
    //
    // Remember to record completion even in case of error.
    //
    //----------------------------------------------------------------

    auto recordCompletion = [&] (stdPars(auto))
    {
        if (kit.dataProcessing)
            require(kit.gpuEventRecording.recordEvent(readingCompletion, kit.gpuCurrentStream, stdPass));

        returnTrue;
    };

    REMEMBER_CLEANUP_EX(errorExit, errorBlock(recordCompletion(stdPassNc)));

    //----------------------------------------------------------------
    //
    // Read image.
    //
    //----------------------------------------------------------------

    require(imageUser(imageSet, image, stdPass));

    //----------------------------------------------------------------
    //
    // Normal completion recording.
    //
    //----------------------------------------------------------------

    errorExit.cancel();
    require(recordCompletion(stdPass));

    ////

    returnTrue;
}

//================================================================
//
// OverlayHolder::setImage
//
//================================================================

template <typename Kit>
stdbool OverlayHolder::setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(Kit))
{
    REQUIRE(size >= 0);

    resetImage();

    //----------------------------------------------------------------
    //
    // Body only for data processing:
    // Currently does not supporting allocations inside the provider(!).
    //
    //----------------------------------------------------------------

    if_not (kit.dataProcessing)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Events.
    //
    //----------------------------------------------------------------

    require(eventsEnsureAlloc(stdPass));

    //----------------------------------------------------------------
    //
    // Get alloc configuration.
    //
    //----------------------------------------------------------------

    Space pitch{};
    require(getAlignedBufferPitch(sizeof(Type), size.X, kit.gpuProperties.samplerRowAlignment, pitch, stdPass));

    ////

    Space allocSize = 0;
    REQUIRE(safeMul(pitch, size.Y, allocSize));

    ////

    constexpr Space maxAllocSize = TYPE_MAX(Space) / sizeof(Type);
    REQUIRE(allocSize <= maxAllocSize);
    Space allocSizeInBytes = allocSize * sizeof(Type);

    //----------------------------------------------------------------
    //
    // Greedy realloc.
    //
    // Basic GPU alloc/free are very slow and synchronous:
    // waiting for all kernels to complete.
    //
    //----------------------------------------------------------------

    if_not (SpaceU(allocSizeInBytes) <= buffer.size())
    {
        require(buffer.realloc(allocSizeInBytes, kit.gpuProperties.samplerAndFastTransferBaseAlignment, stdPass));
    }

    ////

    GpuMatrix<Type> storedImage;
    storedImage.assignUnsafe(buffer.ptr<Type>(), pitch, size.X, size.Y);

    //----------------------------------------------------------------
    //
    // Before using the buffer insert a barrier into our stream
    // to wait for the completion of the other stream buffer usage.
    //
    //----------------------------------------------------------------

    require(kit.gpuEventRecording.putEventDependency(readingCompletion, kit.gpuCurrentStream, stdPass));

    //----------------------------------------------------------------
    //
    // Remember to record completion even in case of error.
    //
    //----------------------------------------------------------------

    auto recordCompletion = [&] (stdPars(auto))
    {
        if (kit.dataProcessing)
            require(kit.gpuEventRecording.recordEvent(writingCompletion, kit.gpuCurrentStream, stdPass));

        returnTrue;
    };

    REMEMBER_CLEANUP_EX(errorExit, errorBlock(recordCompletion(stdPassNc)));

    //----------------------------------------------------------------
    //
    // Overwrite the buffer.
    //
    //----------------------------------------------------------------

    require(provider.saveImage(storedImage, stdPass));

    //----------------------------------------------------------------
    //
    // Normal completion recording.
    //
    //----------------------------------------------------------------

    errorExit.cancel();
    require(recordCompletion(stdPass));

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    image = storedImage;
    imageSet = true;

    returnTrue;
}

//================================================================
//
// OverlayBufferImpl
//
//================================================================

struct OverlayBufferImpl : public OverlayBuffer
{
    //----------------------------------------------------------------
    //
    // Clear memory.
    //
    //----------------------------------------------------------------

    virtual void clearMemory()
    {
        reset();
        holder.dealloc();
    }

    //----------------------------------------------------------------
    //
    // Buffer API.
    //
    //----------------------------------------------------------------

    virtual bool hasUpdates() const
    {
        return cleared || holder.hasImage();
    }

    virtual void reset()
    {
        cleared = false;
        holder.resetImage();
    }

    virtual bool absorb(OverlayBuffer& other)
    {
        auto& that = (OverlayBufferImpl&) other;

        if (that.hasUpdates())
            moveFrom(other);

        return true;
    }

    virtual void moveFrom(OverlayBuffer& other)
    {
        auto& that = (OverlayBufferImpl&) other;

        exchange(holder, that.holder);
        cleared = that.cleared;

        that.reset();
    }

    //----------------------------------------------------------------
    //
    // Data API.
    //
    //----------------------------------------------------------------

    virtual void clearImage()
    {
        holder.resetImage();
        cleared = true;
    }

    virtual stdbool setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(SetImageKit))
    {
        reset();
        return holder.setImage(size, provider, stdPass);
    }

    virtual stdbool useImage(ImageUser& imageUser, stdPars(UseImageKit))
    {
        return holder.useImage(imageUser, stdPass);
    }

    //----------------------------------------------------------------
    //
    // State.
    //
    // States: NONE, CLEARED, IMAGE.
    //
    //----------------------------------------------------------------

    bool cleared = false;
    OverlayHolder holder;

};

////

UniquePtr<OverlayBuffer> OverlayBuffer::create() {return makeUnique<OverlayBufferImpl>();}

//----------------------------------------------------------------

}
