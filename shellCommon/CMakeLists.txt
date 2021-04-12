#================================================================
#
# Common app implementations:
# * Depends on C++ runtime.
# * Depends on STL.
# * Not shell-specific, valid both for GUI and console shells.
#
#================================================================

hexlibProjectTemplate(shellCommon STATIC . "imgBase;imgLib;moduleTools;extLib" FALSE hexlib)

###

if (DEFINED TRACV_PLATFORM_BUILD)
    target_link_libraries(shellCommon PUBLIC tracv::interfaces)
endif()
