#pragma once

#include "channels/buffers/small/boolRequestBuffer.h"

//================================================================
//
// ShutdownBuffer
//
//================================================================

using ShutdownBuffer = BoolRequestBuffer<0xDB16F526u>;
