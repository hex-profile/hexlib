#pragma once

//----------------------------------------------------------------

#if !(defined(TRACV_PLATFORM_BUILD) || defined(TRACV_MODULE_NAME))

    #include "debugBridgeApi.h"

#else

    #include <tracv/debug_tools/debug_tools.hpp>
    namespace debugBridge {using namespace tracv::debug_tools;}

#endif
    
//----------------------------------------------------------------

using debugBridge::DebugBridge;
using debugBridge::DebugBridgeNull;
