#pragma once

#include "dbgptr/dbgptrProtos.h"

//================================================================
//
// Select debug pointers implementation.
//
//================================================================

#if HEXLIB_GUARDED_MEMORY

    #include "dbgptrArrayPointer.h"
    #include "dbgptrMatrixPointer.h"

    #define ArrayPtr(Type) \
        DebugArrayPointer<Type>

    #define ArrayPtrCreate(Type, memPtr, memSize, preconditions) \
        DebugArrayPointer< Type >(memPtr, memSize, preconditions)

    ////

    #define MatrixPtr(Type) \
        DebugMatrixPointer< Type >

    #define MatrixPtrCreate(Type, memPtr, memPitch, memSizeX, memSizeY, preconditions) \
        DebugMatrixPointer< Type >(memPtr, memPitch, memSizeX, memSizeY, preconditions)

#else

    #define ArrayPtr(Element) \
        Element*

    #define ArrayPtrCreate(Element, memPtr, memSize, preconditions) \
        (memPtr)

    ////

    #define MatrixPtr(Element) \
        Element*

    #define MatrixPtrCreate(Element, memPtr, memPitch, memSizeX, memSizeY, preconditions) \
        (memPtr)

#endif
