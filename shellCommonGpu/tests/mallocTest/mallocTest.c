#include "mallocTest.h"

#include "cfgTools/multiSwitch.h"
#include "userOutput/printMsgEx.h"
#include "stl/stlArray.h"
#include "errorLog/errorLog.h"
#include "rndgen/rndgenFloat.h"
#include "dataAlloc/arrayMemory.h"

namespace mallocTest {

//================================================================
//
// MallocChunk
//
//================================================================

class MallocChunk
{

public:

    MallocChunk() =default;

    MallocChunk(size_t n)
    {
        ptr = malloc(n);
        size = n;
    }

    ~MallocChunk()
    {
        if (ptr) free(ptr);
    }

    MallocChunk(MallocChunk&& that)
    {
        ptr = that.ptr;
        size = that.size;

        that.ptr = nullptr;
        that.size = 0;
    }

    MallocChunk& operator =(MallocChunk&& that)
    {
        if (ptr) free(ptr);

        ptr = that.ptr;
        size = that.size;

        that.ptr = nullptr;
        that.size = 0;

        return *this;
    }

public:

    void* ptr = nullptr;
    size_t size = 0;

};

//================================================================
//
// MallocTestImpl
//
//================================================================

class MallocTestImpl : public MallocTest
{

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != Display::Nothing;}
    stdbool process(stdPars(GpuModuleProcessKit));

private:

    enum class Display {Nothing, MallocTest, COUNT};
    ExclusiveMultiSwitch<Display, Display::COUNT, 0x325E0ECD> displaySwitch;

    StlArray<MallocChunk> pool;
    NumericVar<int32> poolSizeCfg{1, 1 << 30, 16384};
    size_t poolAmount = 0;

    NumericVar<Space> blockMinCfg{1, spaceMax, 8};
    NumericVar<Space> blockMaxCfg{1, spaceMax, 4 * 1024 * 1024};

    RndgenState rndgen = 1;

};

//----------------------------------------------------------------

UniquePtr<MallocTest> MallocTest::create()
{
    return makeUnique<MallocTestImpl>();
}

//================================================================
//
// MallocTestImpl::serialize
//
//================================================================

void MallocTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize(kit, STR("Display"), {STR("<Nothing>"), STR("")}, {STR("Malloc Test")});
    poolSizeCfg.serialize(kit, STR("Pool Size"));
    blockMinCfg.serialize(kit, STR("Block Size Min"));
    blockMaxCfg.serialize(kit, STR("Block Size Max"));
}

//================================================================
//
// MallocTestImpl::process
//
//================================================================

stdbool MallocTestImpl::process(stdPars(GpuModuleProcessKit))
{
    Display displayType = kit.verbosity >= Verbosity::On ? displaySwitch : Display::Nothing;

    if (displayType == Display::Nothing)
        returnTrue;

    ////

    printMsgL(kit, STR("Malloc test, consuming % Mb"), fltf(ldexpf(float32(poolAmount), -20), 2));

    //----------------------------------------------------------------
    //
    // Size generation.
    //
    //----------------------------------------------------------------

    auto generateSize = [&] ()
    {
        auto sizeMin = minv(blockMinCfg(), blockMaxCfg());
        auto sizeMax = maxv(blockMinCfg(), blockMaxCfg());

        auto r = rndgenUniform<float32>(rndgen);
        auto sizef = sizeMin * powf(float32(sizeMax) / float32(sizeMin), r);
        auto size = convertNearest<Space>(sizef);
        return clampRange(size, sizeMin, sizeMax);
    };

    //----------------------------------------------------------------
    //
    // Pool resetup.
    //
    //----------------------------------------------------------------

    Space poolSize = poolSizeCfg;

    if_not (pool.size() == poolSize)
    {
        printMsgL(kit, STR("Pool re-setup!"), msgWarn);

        require(pool.realloc(poolSize, stdPass));
        poolAmount = 0;

        ARRAY_EXPOSE(pool);

        for_count (i, poolSize)
        {
            auto& p = poolPtr[i];

            auto size = generateSize();
            p = MallocChunk(size);
            REQUIRE(p.ptr != 0);

            poolAmount += size;
        }
    }

    //----------------------------------------------------------------
    //
    // Prepare block sizes.
    //
    //----------------------------------------------------------------

    ARRAY_ALLOC(blockSize, Space, poolSize);

    if (kit.dataProcessing)
    {
        for_count (i, poolSize)
            blockSize[i] = generateSize();
    }

    //----------------------------------------------------------------
    //
    // Measured part.
    //
    //----------------------------------------------------------------

    if (kit.dataProcessing)
    {
        stdEnterEx(TRACE_AUTO_LOCATION, poolSize, STR("Malloc"));

        ARRAY_EXPOSE(pool);
        ARRAY_EXPOSE(blockSize);

        for_count (i, poolSize)
        {
            auto& f = poolPtr[i];

            poolAmount -= f.size;

            f = MallocChunk(blockSizePtr[i]);
            REQUIRE(f.ptr != 0);

            poolAmount += f.size;
        }
    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
