#pragma once

#include "numbers/divRoundCompile.h"

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
    extern "C" __global__ void func(Params params)

#define hostDeclareKernel(func, Params, params)

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

#define devDecl \
    __device__

#define devPars \
    int

#define devPass \
    0

//================================================================
//
// devConstant
//
//================================================================

#define devConstant \
    __constant__

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

#define devThreadX Space(threadIdx.x)
#define devThreadY Space(threadIdx.y)
#define devThreadIdx  point(devThreadX, devThreadY)

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

#define devThreadCountX Space(blockDim.x)
#define devThreadCountY Space(blockDim.y)
#define devThreadCount  point(devThreadCountX, devThreadCountY)

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

#define devGroupX Space(blockIdx.x)
#define devGroupY Space(blockIdx.y)
#define devGroupZ Space(blockIdx.z)

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

#define devGroupCountX Space(gridDim.x)
#define devGroupCountY Space(gridDim.y)
#define devGroupCountZ Space(gridDim.z)

#define devGroupCount point(devGroupCountX, devGroupCountY)
#define devGroupCount3D point3D(devGroupCountX, devGroupCountY, devGroupCountZ)

//================================================================
//
// devSyncThreads
//
// Synchronizes all threads in the group
//
//================================================================

#define devSyncThreads() \
    __syncthreads()

//================================================================
//
// devDebugCheck
//
// Checks a condition, only in emulation mode.
//
//================================================================

#define devDebugCheck(condition) \
    ((void) (condition))

//================================================================
//
// devAbortCheck
//
// Checks a condition and aborts kernel if it is false.
//
//================================================================

#define devAbortCheck(condition) \
    if (allv(condition)) ; else asm("trap;")

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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 200 || __CUDA_ARCH__ == 350 || __CUDA_ARCH__ == 500 || __CUDA_ARCH__ == 600)

    #define devWarpSize 32
    #define devSramBankPeriod (32*4)

#else

    #error

#endif

//================================================================
//
// devSramVar
//
// Defines local variable in SRAM memory.
//
//================================================================

#define devSramVar(name, Type) \
    __shared__ Type name;

//================================================================
//
// devSramArray
//
// Defines local array in SRAM memory.
//
//================================================================

#define devSramArray(name, Type, size) \
    \
    __shared__ Type name[size]; \
    enum {name##Size = size};

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
    __shared__ Type name##MemPtr[(memPitch) * (sizeY)]; MAKE_VARIABLE_USED(name##MemPtr); \
    enum {name##MemPitch = (memPitch)};

#define devSramMatrixDense(name, Type, sizeX, sizeY) \
    devSramMatrixEx(name, Type, sizeX, sizeY, sizeX)

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

#define devUnrollLoop __pragma(unroll)

//================================================================
//
// devWarpAll
//
// Warp vote functions, to avoid branch divergence.
//
//================================================================

#define devWarpAll(cond) __all(cond)
