#pragma once

//================================================================
//
// DebugBridge
//
//================================================================

#if !(defined(TRACV_PLATFORM_BUILD) || defined(TRACV_MODULE_NAME))

    namespace debugBridge {struct DebugBridge;}

#else

    namespace tracv::debug_tools {struct DebugBridge;}
    namespace debugBridge {using namespace tracv::debug_tools;}

#endif

//----------------------------------------------------------------

using debugBridge::DebugBridge;
