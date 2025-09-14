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
    void setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(Kit));

    ////

    template <typename Kit>
    void useImage(ImageUser& imageUser, stdPars(Kit));

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
    void eventsEnsureAlloc(stdPars(Kit))
    {
        if (eventsAllocated)
            return;

        ////

        REMEMBER_CLEANUP_EX(errorCleanup, eventsDealloc());

        kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, readingCompletion, stdPass);
        kit.gpuEventAlloc.eventCreate(kit.gpuCurrentContext, false, writingCompletion, stdPass);

        ////

        errorCleanup.cancel();
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
void OverlayHolder::useImage(ImageUser& imageUser, stdPars(Kit))
{

    //----------------------------------------------------------------
    //
    // If there is no image, use a simple path without data access.
    //
    //----------------------------------------------------------------

    if_not (imageSet)
    {
        imageUser(false, {}, stdPass);
        return;
    }

    //----------------------------------------------------------------
    //
    // Before using the buffer insert a barrier into our stream
    // to wait for the completion of the other stream buffer usage.
    //
    //----------------------------------------------------------------

    if (kit.dataProcessing)
        kit.gpuEventRecording.putEventDependency(writingCompletion, kit.gpuCurrentStream, stdPass);

    //----------------------------------------------------------------
    //
    // Remember to record completion even in case of error.
    //
    //----------------------------------------------------------------

    auto recordCompletion = [&] (stdPars(auto))
    {
        if (kit.dataProcessing)
            kit.gpuEventRecording.recordEvent(readingCompletion, kit.gpuCurrentStream, stdPass);
    };

    REMEMBER_CLEANUP_EX(errorExit, errorBlock(recordCompletion(stdPassNc)));

    //----------------------------------------------------------------
    //
    // Read image.
    //
    //----------------------------------------------------------------

    imageUser(imageSet, image, stdPass);

    //----------------------------------------------------------------
    //
    // Normal completion recording.
    //
    //----------------------------------------------------------------

    errorExit.cancel();
    recordCompletion(stdPass);
}

//================================================================
//
// OverlayHolder::setImage
//
//================================================================

template <typename Kit>
void OverlayHolder::setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(Kit))
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
        return;

    //----------------------------------------------------------------
    //
    // Events.
    //
    //----------------------------------------------------------------

    eventsEnsureAlloc(stdPass);

    //----------------------------------------------------------------
    //
    // Get alloc configuration.
    //
    //----------------------------------------------------------------

    Space pitch{};
    getAlignedBufferPitch(sizeof(Type), size.X, kit.gpuProperties.samplerRowAlignment, pitch, stdPass);

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
        buffer.realloc(allocSizeInBytes, kit.gpuProperties.samplerAndFastTransferBaseAlignment, stdPass);
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

    kit.gpuEventRecording.putEventDependency(readingCompletion, kit.gpuCurrentStream, stdPass);

    //----------------------------------------------------------------
    //
    // Remember to record completion even in case of error.
    //
    //----------------------------------------------------------------

    auto recordCompletion = [&] (stdPars(auto))
    {
        if (kit.dataProcessing)
            kit.gpuEventRecording.recordEvent(writingCompletion, kit.gpuCurrentStream, stdPass);
    };

    REMEMBER_CLEANUP_EX(errorExit, errorBlock(recordCompletion(stdPassNc)));

    //----------------------------------------------------------------
    //
    // Overwrite the buffer.
    //
    //----------------------------------------------------------------

    provider.saveImage(storedImage, stdPass);

    //----------------------------------------------------------------
    //
    // Normal completion recording.
    //
    //----------------------------------------------------------------

    errorExit.cancel();
    recordCompletion(stdPass);

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    image = storedImage;
    imageSet = true;
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

    virtual void setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(SetImageKit))
    {
        reset();
        holder.setImage(size, provider, stdPass);
    }

    virtual void useImage(ImageUser& imageUser, stdPars(UseImageKit))
    {
        holder.useImage(imageUser, stdPass);
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
