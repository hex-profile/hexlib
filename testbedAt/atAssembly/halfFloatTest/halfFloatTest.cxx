#if HOSTCODE
#include "halfFloatTest.h"
#endif

#include "gpuSupport/gpuTool.h"
#include "vectorTypes/half/halfType.h"
#include "rndgen/rndgenBase.h"

#if HOSTCODE
#include <memory.h>
#include "storage/classThunks.h"
#include "dataAlloc/arrayMemory.inl"
#include "dataAlloc/gpuArrayMemory.h"
#include "userOutput/printMsgEx.h"
#endif

namespace halfFloatTest {

//================================================================
//
// genSequentialPackTestFloats
//
//================================================================

GPUTOOL_2D
(
    genSequentialPackTestFloats,
    PREP_EMPTY,
    ((float32, dst)),
    ((uint32, low16bits)),
  
    {
        uint32 index = devGroupX * devThreadCountX + devThreadX;
        uint32 testValue = (index << 16) | low16bits;
        (uint32&) helpModify(*dst) = testValue;
    }
)

//================================================================
//
// packHalfGpu
//
//================================================================

GPUTOOL_2D
(
    packToHalfGpu,
    PREP_EMPTY,
    ((const float32, src))
    ((float16, dst)),
    PREP_EMPTY,
    *dst = packFloat16(*src);
)

//================================================================
//
// genSequentialUnpackTestFloats
//
//================================================================

GPUTOOL_2D
(
    genSequentialUnpackTestFloats,
    PREP_EMPTY,
    ((float16, dst)),
    PREP_EMPTY,
  
    {
        uint32 index = devGroupX * devThreadCountX + devThreadX;

        float16 tmp;
        tmp.data = index;

        *dst = tmp;
    }
)

//================================================================
//
// unpackHalfGpu
//
//================================================================

GPUTOOL_2D
(
    unpackHalfGpu,
    PREP_EMPTY,
    ((const float16, src))
    ((uint32, dst)),
    PREP_EMPTY,
    (float32&) helpModify(*dst) = unpackFloat16(*src);
)

//================================================================
//
// 
//
//================================================================

#if HOSTCODE

//================================================================
//
// packToHalfCpu
//
//================================================================

bool packToHalfCpu(const Array<const float32>& src, const Array<float16>& dst, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true;

    ////

    REQUIRE(equalSize(src, dst));
    Space size = src.size();

    ARRAY_EXPOSE(src);
    ARRAY_EXPOSE(dst);

    #pragma omp parallel for

    for (Space i = 0; i < size; ++i)
    {
        dstPtr[i] = packFloat16(helpRead(srcPtr[i]));
    }

    ////

    stdEnd;
}

//================================================================
//
// unpackToHalfCpu
//
//================================================================

bool unpackToHalfCpu(const Array<const float16>& src, const Array<uint32>& dst, stdPars(CpuFuncKit))
{
    stdBegin;

    if_not (kit.dataProcessing)
        return true;

    ////

    REQUIRE(equalSize(src, dst));
    Space size = src.size();

    ARRAY_EXPOSE(src);
    ARRAY_EXPOSE(dst);

    #pragma omp parallel for

    for (Space i = 0; i < size; ++i)
    {
        (float32&) helpModify(dstPtr[i]) = unpackFloat16(helpRead(srcPtr[i]));
    }

    ////

    stdEnd;
}

//================================================================
//
// operator ==(float16)
//
//================================================================

sysinline bool operator ==(const float16& A, const float16& B)
{
    return A.data == B.data;
}

//================================================================
//
// compareResults
//
//================================================================

template <typename Type>
bool compareResults(const Array<const Type>& ref, const Array<const Type>& tst, Space& badIndex, stdPars(CpuFuncKit))
{
    stdBegin;
  
    if_not (kit.dataProcessing)
        return true;

    ////

    REQUIRE(equalSize(ref, tst));
    Space size = ref.size();

    ////

    ARRAY_EXPOSE(ref);
    ARRAY_EXPOSE(tst);

    ////

    bool allOk = memcmp(unsafePtr(refPtr, size), unsafePtr(tstPtr, size), sizeof(float16) * size) == 0;

    if (allOk)
        return true;

    ////

    Space badIdx = TYPE_MAX(Space);

    #pragma omp parallel for

    for (Space i = 0; i < size; ++i)
    {
        if_not (helpRead(refPtr[i]) == helpRead(tstPtr[i]))
        {
            #pragma omp critical
            {
                badIdx = minv(badIdx, i);
            }
        }
    }

    REQUIRE(badIdx != TYPE_MAX(Space));
    badIndex = badIdx;
    return false;

    stdEnd;
}

//================================================================
//
// HalfFloatTestImpl
//
//================================================================

class HalfFloatTestImpl
{

public:

    void serialize(const ModuleSerializeKit& kit) {}
    bool reallocValid() const {return true;}
    bool realloc(stdPars(ReallocKit));
    bool process(const Process& o, stdPars(ProcessKit));

private:

    bool testPacking(stdPars(ProcessKit));
    bool testUnpacking(stdPars(ProcessKit));

private:

    bool unpackTestFinished = false;

    ////

    uint32 packTestPass = 0;
    bool packTestFinished = false;

};

//================================================================
//
// HalfFloatTestImpl::realloc
//
//================================================================

bool HalfFloatTestImpl::realloc(stdPars(ReallocKit))
{
    stdBegin;
    stdEnd;
}

//================================================================
//
// HalfFloatTestImpl::testPacking
//
//================================================================

bool HalfFloatTestImpl::testPacking(stdPars(ProcessKit))
{
    stdBegin;

    //----------------------------------------------------------------
    //
    // One testing pass
    //
    //----------------------------------------------------------------

    //
    // Generate test floats
    //

    static const Space packTestSize = 1 << 16;

    GpuArrayMemory<float32> src;
    require(src.realloc(packTestSize, stdPass));
    require(genSequentialPackTestFloats(src, packTestPass, stdPass));

    //
    // Copy src to CPU
    //

    ArrayMemory<float32> srcCopy;
    require(srcCopy.reallocForGpuExch(packTestSize, stdPass));

    {
        GpuCopyThunk gpuCopy;
        require(gpuCopy(src, srcCopy, stdPass));
    }

    //
    // Process on GPU
    //
  
    GpuArrayMemory<float16> gpuResult;
    require(gpuResult.realloc(packTestSize, stdPass));
    require(packToHalfGpu(src, gpuResult, stdPass));

    //
    // Copy GPU result to CPU
    //

    ArrayMemory<float16> gpuResultCopy;
    require(gpuResultCopy.reallocForGpuExch(packTestSize, stdPass));

    {
        GpuCopyThunk gpuCopy;
        require(gpuCopy(gpuResult, gpuResultCopy, stdPass));
    }

    //
    // Process on CPU
    //

    ArrayMemory<float16> cpuResult;
    require(cpuResult.reallocForGpuExch(packTestSize, stdPass));

    require(packToHalfCpu(srcCopy, cpuResult, stdPass));

    //
    // Compare results
    //

    Space badIdx = 0;

    if_not (compareResults<float16>(gpuResultCopy, cpuResult, badIdx, stdPass))
    {
        ARRAY_EXPOSE(srcCopy);
        ARRAY_EXPOSE(gpuResultCopy);
        ARRAY_EXPOSE(cpuResult);

        float32 badValue = srcCopyPtr[badIdx];

        printMsgG(kit, STR("Incorrect value: src value %0 (%1), correct packed %2, but received %3, at pass=%4 idx=%5"), 
            hex((uint32&) badValue, 8),
            fltg(badValue, 8),
            hex((uint16&) helpRead(gpuResultCopyPtr[badIdx]), 4),
            hex((uint16&) helpRead(cpuResultPtr[badIdx]), 4),
            packTestPass, badIdx,
            msgErr);

        return false;
    }

    //
    // Advance
    //

    if (kit.dataProcessing)
    {
        if_not (packTestFinished)
        {
            packTestPass++;

            if (packTestPass == 0x10000)
            {
                packTestFinished = true;
                packTestPass = 0;
            }
        }
  }


    ////

    stdEnd;
}

//================================================================
//
// HalfFloatTestImpl::testUnpacking
//
//================================================================

bool HalfFloatTestImpl::testUnpacking(stdPars(ProcessKit))
{
    stdBegin;

    const Space testSize = 1 << 16;

    //----------------------------------------------------------------
    //
    // One testing pass
    //
    //----------------------------------------------------------------

    GpuArrayMemory<float16> src;
    require(src.realloc(testSize, stdPass));
    require(genSequentialUnpackTestFloats(src, stdPass));

    //
    // Copy src to CPU
    //

    ArrayMemory<float16> srcCopy;
    require(srcCopy.reallocForGpuExch(testSize, stdPass));

    {
        GpuCopyThunk gpuCopy;
        require(gpuCopy(src, srcCopy, stdPass));
    }

    //
    // Process on GPU
    //
  
    GpuArrayMemory<uint32> gpuResult;
    require(gpuResult.realloc(testSize, stdPass));
    require(unpackHalfGpu(src, gpuResult, stdPass));

    //
    // Copy GPU result to CPU
    //

    ArrayMemory<uint32> gpuResultCopy;
    require(gpuResultCopy.reallocForGpuExch(testSize, stdPass));

    {
        GpuCopyThunk gpuCopy;
        require(gpuCopy(gpuResult, gpuResultCopy, stdPass));
    }

    //
    // Process on CPU
    //

    ArrayMemory<uint32> cpuResult;
    require(cpuResult.reallocForGpuExch(testSize, stdPass));

    require(unpackToHalfCpu(srcCopy, cpuResult, stdPass));

    //
    // Compare results
    //

    Space badIdx = 0;

    if_not (compareResults<uint32>(gpuResultCopy, cpuResult, badIdx, stdPass))
    {
        ARRAY_EXPOSE(srcCopy);
        ARRAY_EXPOSE(gpuResultCopy);
        ARRAY_EXPOSE(cpuResult);

        float16 badValue = srcCopyPtr[badIdx];

        printMsgG(kit, STR("Incorrect value: src value %0, correct unpacked %1, but received %2, at idx=%3"), 
            hex(badValue.data, 4),
            hex(helpRead(gpuResultCopyPtr[badIdx]), 8),
            hex(helpRead(cpuResultPtr[badIdx]), 8),
            badIdx,
            msgErr);

        return false;
    }

    stdEnd;
}

//================================================================
//
// HalfFloatTestImpl::process
//
//================================================================

bool HalfFloatTestImpl::process(const Process& o, stdPars(ProcessKit))
{
    stdBegin;

    REQUIRE(reallocValid());

    //----------------------------------------------------------------
    //
    // Test unpacking
    //
    //----------------------------------------------------------------

    if (unpackTestFinished)
        printMsgL(kit, STR("Half-float unpack test: PASSED"), msgInfo);
    else
    {
        require(testUnpacking(stdPass));

        if (kit.dataProcessing)
            unpackTestFinished = true;

        return true;
    }

    //----------------------------------------------------------------
    //
    // Test packing
    //
    //----------------------------------------------------------------

    if (packTestFinished)
    {
        printMsgL(kit, STR("Half-float pack test: PASSED"), msgInfo);
    }
    else
    {
        for (int32 k = 0; k < 16; ++k)
            require(testPacking(stdPass));

        printMsgL(kit, STR("Half-float pack test: %0%%"), fltf(float32(packTestPass) / float32(0x10000) * 100, 1), msgInfo);
    }

    stdEnd;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(HalfFloatTest)
CLASSTHUNK_VOID1(HalfFloatTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(HalfFloatTest, reallocValid)
CLASSTHUNK_BOOL_STD0(HalfFloatTest, realloc, ReallocKit)
CLASSTHUNK_BOOL_STD1(HalfFloatTest, process, const Process&, ProcessKit)

//----------------------------------------------------------------

#endif

}
