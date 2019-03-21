#pragma once

#include "storage/opaqueStruct.h"

//================================================================
//
// GpuContext
//
// Concurrent thread usage of a context:
//
// * Concurrect context destroying NOT allowed.
// * Concurrent setThreadContext allowed.
// * Concurrent module create/destroy in the same context allowed.
// * Concurrent CPU/GPU memory alloc/dealloc in the same context allowed.
// * Concurrent texture create/destroy in the same context allowed.
// * Concurrent stream create/destroy in the same context allowed.
// * Concurrent event create/destroy in the same context allowed.
// * Concurrent execution API allowed: setSampler/kernel calling.
//
//================================================================

struct GpuContext : public OpaqueStruct<8> {};

//================================================================
//
// GpuStream
//
// Concurrent thread usage of a stream currently NOT allowed.
//
//================================================================

struct GpuStream : public OpaqueStruct<8> {};

//================================================================
//
// GpuEvent
//
// Concurrent thread access:
// * Concurrent event creation/destruction NOT allowed.
// * Concurrent usage of an event is allowed.
//
//================================================================

struct GpuEvent : public OpaqueStruct<8> {};

//================================================================
//
// GpuTexture
//
// Concurrent thread access currently is NOT allowed.
//
//================================================================

struct GpuTexture : public OpaqueStruct<32> {};
