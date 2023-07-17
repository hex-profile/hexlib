#pragma once

#include "storage/opaqueStruct.h"

//================================================================
//
// GpuContext
//
// Concurrent thread usage of a context:
//
// * Concurrect context destroying NOT allowed.
// * Concurrent threadContextSet allowed.
// * Concurrent module create/destroy in the same context allowed.
// * Concurrent CPU/GPU memory alloc/dealloc in the same context allowed.
// * Concurrent texture create/destroy in the same context allowed.
// * Concurrent stream create/destroy in the same context allowed.
// * Concurrent event create/destroy in the same context allowed.
// * Concurrent execution API allowed: setSampler/kernel calling.
//
//================================================================

using GpuContext = OpaqueStruct<8, 0xAD23E3A0u>;

//================================================================
//
// GpuStream
//
// Concurrent thread usage of a stream currently NOT allowed.
//
//================================================================

using GpuStream = OpaqueStruct<8, 0x98F6A9F0u>;

//================================================================
//
// GpuEvent
//
// Concurrent thread access:
// * Concurrent event creation/destruction NOT allowed.
// * Concurrent usage of an event is allowed.
//
//================================================================

using GpuEvent = OpaqueStruct<8, 0xE06D6391u>;

//================================================================
//
// GpuTexture
//
// Concurrent thread access currently is NOT allowed.
//
//================================================================

using GpuTexture = OpaqueStruct<32, 0x2CCAB6C4u>;
