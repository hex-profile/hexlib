#include "fibers.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

namespace fibers {

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// WIN32
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#if defined(_WIN32)

//================================================================
//
// fiberConvertThreadToFiber
//
//================================================================

bool fiberConvertThreadToFiber(Fiber& fiber)
{
    fiber.handle = ConvertThreadToFiber(0);
    return fiber.handle != nullptr;
}

//================================================================
//
// fiberConvertFiberToThread
//
//================================================================

bool fiberConvertFiberToThread()
{
    return ConvertFiberToThread() != 0;
}

//================================================================
//
// fiberCreate
//
//================================================================

bool fiberCreate(FiberFunc* func, void* param, size_t stackSize, Fiber& fiber)
{
    fiber.handle = CreateFiberEx(0, stackSize, 0, func, param);
    return fiber.handle != nullptr;
}

//================================================================
//
// fiberDestroy
//
//================================================================

void fiberDestroy(Fiber& fiber)
{
    if (fiber.handle != nullptr)
    {
        DeleteFiber(fiber.handle);
        fiber.handle = nullptr;
    }
}

//================================================================
//
// fiberSwitch
//
//================================================================

void fiberSwitch(Fiber& oldFiber, const Fiber& newFiber)
{
    SwitchToFiber(newFiber.handle);
}

//----------------------------------------------------------------

#endif

//----------------------------------------------------------------

}
