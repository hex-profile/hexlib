#pragma once

#include "charType/charType.h"
#include "errorLog/errorLog.h"
#include "kit/kit.h"
#include "data/space.h"
#include "storage/addrSpace.h"
#include "point/point.h"
#include "storage/typeAlignment.h"
#include "point3d/point3dBase.h"
#include "numbers/divRoundCompile.h"

//================================================================
//
// EmuError
//
// 0 means no error, otherwise contains pointer to error message (in static memory).
//
//================================================================

using EmuError = const CharType*;

#define EMU_ERRMSG(condition) \
    (CHECK_TRACE_PREFIX CHECK_FAIL_MSG(condition))

//================================================================
//
// EmuKernelTools
//
// Special interface functions, available inside kernel code.
//
//================================================================

struct EmuKernelTools
{
    virtual void syncThreads(Space fiberIdx, uint32 id, EmuError errMsg) =0;
    virtual void fatalError(EmuError errMsg) =0;
};

//================================================================
//
// EmuSharedParams
//
// Shared emu thread parameters, to save space.
//
//================================================================

KIT_CREATE4(
    EmuSharedParams,
    Point<Space>, threadCount,
    Point3D<Space>, groupIdx,
    Point3D<Space>, groupCount,
    EmuKernelTools&, kernelTools
);

//================================================================
//
// EmuSramAllocator
//
// Very fast SRAM allocator.
// Supports only allocation (no deallocation).
//
//================================================================

class EmuSramAllocator
{

private:

    static const CpuAddrU alignment = maxNaturalAlignment;
    COMPILE_ASSERT(COMPILE_IS_POWER2(alignment));
    COMPILE_ASSERT(alignment >= 1);
    static const CpuAddrU alignmentMask = alignment - 1;

public:

    inline EmuSramAllocator()
    {
        memPtr = 0;
        memEnd = 0;
    }

    //
    // Setup.
    // Both memory address and size should be aligned at maxNaturalAlignment bytes!
    //

    inline void setup(CpuAddrU memAddr, CpuAddrU memSize, int IUnderstandAligmentRequirements)
    {
        this->memPtr = memAddr;
        this->memEnd = memAddr + memSize;
    }

    //
    // Allocate memory. Can return 0!
    //

    inline CpuAddrU alloc(CpuAddrU size)
    {
        // Get the available space (always aligned)
        CpuAddrU availSpace = memEnd - memPtr;

        // Does the unaligned size fit into the available space?
        require(size <= availSpace);

        //
        // The unaligned size fits into the available space AND
        // the available size is aligned ==>
        //
        // * The rounded up size also fits into the available space.
        // * Computing the rounded up size will not cause arithmetic overflow.
        //

        CpuAddrU alignedSize = (size + alignmentMask) & ~alignmentMask;

        // Original memory pointer (for result).
        CpuAddrU prevPtr = memPtr;

        // Record changes.
        memPtr = memPtr + alignedSize;

        // Result: previous aligned pointer.
        return prevPtr;
    }

private:

    //
    // Both memPtr and memEnd are always aligned.
    // memPtr <= memEnd
    //

    CpuAddrU memPtr;
    CpuAddrU memEnd;

};

//================================================================
//
// EmuParams
//
// Contains only the varying part of thread's emu parameters.
//
//================================================================

struct EmuParams
{
    const EmuSharedParams* sharedParams;
    Space fiberIdx;
    Point<Space> threadIdx;
    EmuSramAllocator sramAllocator; // by value
};

//================================================================
//
// EmuKernelFunc
//
//================================================================

typedef void EmuKernelFunc(const void* params, EmuParams& emuParams);

//================================================================
//
// GpuKernelLink
//
//================================================================

struct GpuKernelLink
{
    EmuKernelFunc* func;
};

//================================================================
//
// devDefineKernel
//
// Makes the header of a kernel function.
//
// User specifies the name of the function and
// the type of parameters structure.
//
//================================================================

#define devDefineKernel(func, Params, params) \
    static void PREP_PASTE(func, Code)(const Params& params, EmuParams& emuParams); \
    namespace {const GpuKernelLink func = {(EmuKernelFunc*) PREP_PASTE(func, Code)};} \
    void PREP_PASTE(func, Code)(const Params& params, EmuParams& emuParams)

//----------------------------------------------------------------

#define hostDeclareKernel(func, Params, params) \
    namespace {extern const GpuKernelLink func;}

//================================================================
//
// devDecl
//
// Function modifier that makes the function device-only.
//
// devPars
// devPass
//
// Declare parameters and pass parameters macros
// for top-level (kernel-level) device functions.
//
//================================================================

#define devDecl

#define devPars \
    EmuParams& emuParams

#define devPass \
    emuParams

//================================================================
//
// devConstant
//
//================================================================

#define devConstant

//================================================================
//
// devThreadX
// devThreadY
// devThreadIdx
//
// Get thread index (of type Space).
// devThreadIdx returns 2D value.
//
//================================================================

#define devThreadIdx (emuParams.threadIdx)
#define devThreadX (emuParams.threadIdx.X)
#define devThreadY (emuParams.threadIdx.Y)

//================================================================
//
// devThreadCountX
// devThreadCountY
// devThreadCount
//
// Get the number of threads in the group (of type Space).
// devThreadCount returns 2D value.
//
//================================================================

#define devThreadCount (emuParams.sharedParams->threadCount)
#define devThreadCountX (emuParams.sharedParams->threadCount.X)
#define devThreadCountY (emuParams.sharedParams->threadCount.Y)

//================================================================
//
// devGroupX / devGroupY / devGroupZ
//
// devGroupIdx
// devGroupIdx3D
//
// Get group index (of type Space), devGroupIdx returns 2D value, devGroupIdx3D returns 3D value.
//
//================================================================

#define devGroupX (emuParams.sharedParams->groupIdx.X)
#define devGroupY (emuParams.sharedParams->groupIdx.Y)
#define devGroupZ (emuParams.sharedParams->groupIdx.Z)

#define devGroupIdx point(devGroupX, devGroupY)
#define devGroupIdx3D point3D(devGroupX, devGroupY, devGroupZ)

//================================================================
//
// devGroupCountX / devGroupCountY / devGroupCountZ
//
// devGroupCount
// devGroupCount3D
//
// Get the number of groups (of type Space), devGroupCount returns 2D value, devGroupCount3D returns 3D value.
//
//================================================================

#define devGroupCountX (emuParams.sharedParams->groupCount.X)
#define devGroupCountY (emuParams.sharedParams->groupCount.Y)
#define devGroupCountZ (emuParams.sharedParams->groupCount.Z)

#define devGroupCount point(devGroupCountX, devGroupCountY)
#define devGroupCount3D (emuParams.sharedParams->groupCount)

//================================================================
//
// devSyncThreads
//
// Synchronizes all threads in the group.
//
//================================================================

#define devSyncThreads() \
    emuParams.sharedParams->kernelTools.syncThreads(emuParams.fiberIdx, __LINE__, EMU_ERRMSG(devSyncThreads()))

//================================================================
//
// devDebugCheck
//
// Checks a condition, only in emulation mode.
//
//================================================================

#define devDebugCheck(condition) \
    if (allv(condition)) ; else throw EMU_ERRMSG(condition)

//================================================================
//
// devAbortCheck
//
// Checks a condition and aborts kernel if it is false.
//
//================================================================

#define devAbortCheck(condition) \
    if (allv(condition)) ; else throw EMU_ERRMSG(condition)

//================================================================
//
// devWarpSize
//
// Execution unit SIMD size
//
// devSramBankPeriod
//
// Gives SRAM bank period IN BYTES.
//
//================================================================

#define devWarpSize 32
#define devSramBankPeriod (32*4)

//================================================================
//
// devSramVar
//
// Defines local variable in SRAM memory.
//
//================================================================

#define devSramVar(name, Type) \
    CpuAddrU name##Addr = emuParams.sramAllocator.alloc(sizeof(Type)); \
    devDebugCheck(name##Addr != 0); \
    Type& name = * (Type*) name##Addr;

//================================================================
//
// devSramArray
//
// Defines local array in SRAM memory.
//
//================================================================

#define devSramArray(name, Type, size) \
    CpuAddrU PREP_PASTE(name, Addr) = emuParams.sramAllocator.alloc((size) * sizeof(Type)); \
    devDebugCheck(PREP_PASTE(name, Addr) != 0); \
    ArrayPtr(Type) name = ArrayPtrCreate(Type, (Type*) PREP_PASTE(name, Addr), (size), DbgptrArrayPreconditions())

//================================================================
//
// devSramMatrixEx
// devSramMatrixDense
//
// Defines local matrix in SRAM memory.
//
//================================================================

#define devSramMatrixEx(name, Type, sizeX, sizeY, memPitch) \
    \
    COMPILE_ASSERT((sizeX) >= 0 && (sizeY) >= 0); \
    COMPILE_ASSERT((sizeX) <= (memPitch)); \
    \
    enum {name##MemPitch = (memPitch)}; \
    CpuAddrU name##Addr = emuParams.sramAllocator.alloc(name##MemPitch * (sizeY) * sizeof(Type)); \
    devDebugCheck(name##Addr != 0); \
    MatrixPtr(Type) name##MemPtr = MatrixPtrCreate(Type, (Type*) name##Addr, name##MemPitch, sizeX, sizeY, DbgptrMatrixPreconditions())

#define devSramMatrixDense(name, Type, sizeX, sizeY) \
    devSramMatrixEx(name, Type, sizeX, sizeY, (sizeX) + 1) /* Extra elements for additonal access checking */

//================================================================
//
// devSramMatrixFor2dAccess
//
// Local matrix in SRAM memory, with pitch chosen to avoid bank conflicts
// for 2D thread group access when a single warp access covers several rows.
//
//----------------------------------------------------------------
//
// Inside a row, the access is dense, so we don't have bank conflicts
// on Fermi+.
//
// Between rows: Select pitch so that the beginning of the
// next row access continues the end of previous row access, modulo
// bank period (32 words for Fermi).
//
// memPitch = 32 K + cacheAccessWidth
//
//================================================================

template <Space sizeX, Space threadCountX, Space typeSize>
struct GpuSramPitchFor2dAccess
{
    COMPILE_ASSERT(sizeX >= 0 && threadCountX >= 0);
    COMPILE_ASSERT(COMPILE_IS_POWER2(typeSize) && typeSize >= 1 && typeSize <= 16);

    COMPILE_ASSERT(devSramBankPeriod % typeSize == 0);
    static const Space bankModulo = devSramBankPeriod / typeSize;

    static const Space resultPitch = ALIGN_UP_NONNEG(COMPILE_CLAMP_MIN(sizeX - threadCountX, 0), bankModulo) + threadCountX;
};

//----------------------------------------------------------------

#define devSramMatrixFor2dAccess(name, Type, sizeX, sizeY, threadCountX) \
    devSramMatrixEx(name, Type, sizeX, sizeY, (GpuSramPitchFor2dAccess<sizeX, threadCountX, sizeof(Type)>::resultPitch))

//================================================================
//
// devUnrollLoop
//
// Tells the compiler to unroll the loop.
//
//================================================================

#define devUnrollLoop

//================================================================
//
// devWarpAll
//
// Warp vote functions, to avoid branch divergence.
//
//================================================================

#define devWarpAll(cond) (cond)
