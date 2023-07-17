#pragma once

//----------------------------------------------------------------

#if defined(TRACV_PLATFORM_BUILD) || defined(TRACV_MODULE_NAME)

    #include <tracv/debug_tools/debug_tools.hpp>
    namespace debugBridge {using namespace tracv::debug_tools;}

#elif defined(RVISION_PLATFORM_BUILD)

    #include <debug_tools/debug_tools.hpp>
    namespace debugBridge {using namespace rvision::debug_tools;}

#else

    #include "debugBridgeApi.h"

#endif

//----------------------------------------------------------------

using debugBridge::DebugBridge;
using debugBridge::DebugBridgeNull;
