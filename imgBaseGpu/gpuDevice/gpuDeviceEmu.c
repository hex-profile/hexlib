#include "gpuDeviceEmu.h"

//================================================================
//
// emuThrowError
//
//================================================================

[[noreturn]]
void emuThrowError(EmuError error)
{
    throw error;
}
