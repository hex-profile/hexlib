include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Common app implementations:
# * Depends on C++ runtime.
# * Depends on STL.
# * Not shell-specific, valid both for GUI and console shells.
#
#================================================================

hexlibProjectTemplate(shellCommon STATIC . "imgBase;imgLib;moduleTools" FALSE hexlib)
