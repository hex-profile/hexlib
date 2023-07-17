#pragma once

//================================================================
//
// DebugBridge
//
//================================================================

#if defined(TRACV_PLATFORM_BUILD) || defined(TRACV_MODULE_NAME)

    namespace tracv::debug_tools {struct DebugBridge;}
    namespace debugBridge {using namespace tracv::debug_tools;}

#elif defined(RVISION_PLATFORM_BUILD)

    namespace rvision::debug_tools {struct DebugBridge;}
    namespace debugBridge {using namespace rvision::debug_tools;}

#else

    namespace debugBridge {struct DebugBridge;}

#endif

//----------------------------------------------------------------

using debugBridge::DebugBridge;
