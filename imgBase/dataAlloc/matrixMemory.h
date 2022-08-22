#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "dataAlloc/cpuDefaultAlignments.h"
#include "dataAlloc/memoryAllocator.h"
#include "data/matrix.h"

//================================================================
//
// MatrixMemory<Type>
//
// Allocates/frees matrix memory.
// Does NOT call element constructors and destructors.
//
// USAGE EXAMPLES:
//
//================================================================

#if 0

// Construct empty matrix; no memory allocation performed.
MatrixMemory<int> m0;

// Allocate matrix; check allocation error.
// If reallocation fails, matrix will have zero size.
// Destructor deallocates memory automatically.
require(m0.realloc(point(33, 17), stdPass));

// Change matrix layout without reallocation; check error.
// New size should be <= allocated size, otherwise the call fails and the layout is not changed.
REQUIRE(m0.resize(point(13, 15)));

// Get current allocated size.
REQUIRE(m0.maxSizeX() == 13);
REQUIRE(m0.maxSizeY() == 15);
REQUIRE(m0.maxSize() == point(13, 15));

// Reallocate matrix base aligned to 512 bytes and pitch aligned to 32 bytes;
// The pitch alignment will be used across "resize" calls until the next "realloc";
require(m0.realloc(point(333, 111), 512, 32, stdPass));
REQUIRE(m0.resize(129, 15));

// Convert to Matrix<> implicitly and explicitly (for template arguments).
Matrix<int> tmp0 = m0;
Matrix<const int> tmp1 = m0;
Matrix<const int> tmp2 = m0();

#endif

//================================================================
//
// MatrixMemoryEx<Pointer>
//
// Flexible implementation for any address space.
//
//================================================================

template <typename Pointer>
class MatrixMemoryEx : public MatrixEx<Pointer>
{

    using AddrU = typename PtrAddrType<Pointer>::AddrU;

    using BaseMatrix = MatrixEx<Pointer>;

public:

    inline MatrixMemoryEx()
        {initZero();}

    inline ~MatrixMemoryEx()
        {dealloc();}

private:

    MatrixMemoryEx(const MatrixMemoryEx<Pointer>& that); // forbidden
    void operator =(const MatrixMemoryEx<Pointer>& that); // forbidden

public:

    stdbool realloc(const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit));

    ////

    void dealloc();

    ////

    inline bool allocated() const {return allocPtr != Pointer(0);}

    ////

    inline Space maxSizeX() const {return allocSize.X;}
    inline Space maxSizeY() const {return allocSize.Y;}
    inline Point<Space> maxSize() const {return allocSize;}

    ////

    inline void resizeNull()
        {BaseMatrix::assignNull();}

    bool resize(Space sizeX, Space sizeY); // rearrange without reallocation

    inline bool resize(const Point<Space>& size)
        {return resize(size.X, size.Y);}

private:

    inline void initZero()
    {
        allocPtr = Pointer(0);
        allocSize = point(0);
        allocAlignMask = 0;
    }

private:

    //
    MemoryOwner memoryOwner;

    // Only for resize support
    Pointer allocPtr;
    Point<Space> allocSize;
    Space allocAlignMask;

};

//================================================================
//
// MatrixMemory<Type>
//
// Allocates/frees matrix memory.
// Does NOT call element constructors and destructors.
//
// C-based address space.
//
//================================================================

template <typename Type>
class MatrixMemory : public MatrixMemoryEx<Type*>
{

    using Base = MatrixMemoryEx<Type*>;

public:

    //
    // Export cast (no code generated, reinterpret 'this')
    //

    sysinline operator const Matrix<Type>& () const
    {
        const MatrixEx<Type*>* baseMatrix = this;
        return recastEqualLayout<const Matrix<Type>>(*baseMatrix);
    }

    sysinline operator const Matrix<const Type>& () const
    {
        const MatrixEx<Type*>* baseMatrix = this;
        return recastEqualLayout<const Matrix<const Type>>(*baseMatrix);
    }

    sysinline const Matrix<Type>& operator()() const
    {
        const MatrixEx<Type*>* baseMatrix = this;
        return recastEqualLayout<const Matrix<Type>>(*baseMatrix);
    }

    //
    // Default realloc: assumes kit.cpuFastAlloc
    //

    using Base::realloc;

    template <typename Kit>
    inline stdbool realloc(const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, stdPars(Kit))
        {return Base::realloc(size, baseByteAlignment, rowByteAlignment, kit.cpuFastAlloc, stdPassThru);}

    template <typename Kit>
    inline stdbool reallocForGpuExch(const Point<Space>& size, stdPars(Kit))
        {return Base::realloc(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.cpuFastAlloc, stdPassThru);}

    template <typename Kit>
    inline stdbool reallocForCpuOnly(const Point<Space>& size, stdPars(Kit))
        {return Base::realloc(size, cpuBaseByteAlignment, cpuRowByteAlignment, kit.cpuFastAlloc, stdPassThru);}

};

//================================================================
//
// MATRIX_ALLOC_FOR_GPU_EXCH
//
//================================================================

#define MATRIX_ALLOC_FOR_GPU_EXCH(name, Type, size) \
    MatrixMemory<Type> name; \
    require(name.reallocForGpuExch(size, stdPass))

#define MATRIX_ALLOC_FOR_CPU_ONLY(name, Type, size) \
    MatrixMemory<Type> name; \
    require(name.reallocForCpuOnly(size, stdPass))
