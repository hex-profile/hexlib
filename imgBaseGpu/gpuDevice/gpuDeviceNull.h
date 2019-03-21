#pragma once

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
    static void func##__unused(Params params)

//----------------------------------------------------------------

#define hostDeclareKernel(func, Params, params) \

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
    int

#define devPass \
    0

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

#define devThreadX 0
#define devThreadY 0
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

#define devThreadCountX 1
#define devThreadCountY 1
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

#define devGroupX 0
#define devGroupY 0
#define devGroupZ 0
#define devGroupIdx point(devGroupX, devGroupY)

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

#define devGroupCountX 1
#define devGroupCountY 1
#define devGroupCountZ 1

#define devGroupCount point(devGroupCountX, devGroupCountY)
#define devGroupCount3D point(devGroupCountX, devGroupCountY, devGroupCountZ)

//================================================================
//
// devSyncThreads
//
// Synchronizes all threads in the group.
//
//================================================================

#define devSyncThreads()

//================================================================
//
// devDebugCheck
//
// Checks a condition, only in emulation mode.
//
//================================================================

#define devDebugCheck(condition)

//================================================================
//
// devAbortCheck
//
// Checks a condition and aborts kernel if it is false.
//
//================================================================

#define devAbortCheck(condition)

//================================================================
//
// devWarpSize
// devSramBankPeriod
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
    Type name;

//================================================================
//
// devSramArray
//
// Defines local array in SRAM memory.
//
//================================================================

#define devSramArray(name, Type, size) \
    \
    Type name[size]; \
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
    Type name##MemPtr[(memPitch) * (sizeY)]; \
    enum {name##MemPitch = (memPitch)};

#define devSramMatrixDense(name, Type, sizeX, sizeY) \
    devSramMatrixEx(name, Type, sizeX, sizeY, sizeX)

//================================================================
//
// devSramMatrixFor2dAccess
//
//================================================================

#define devSramMatrixFor2dAccess(name, Type, sizeX, sizeY, threadCountX) \
    devSramMatrixDense(name, Type, sizeX, sizeY)

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
