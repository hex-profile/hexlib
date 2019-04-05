include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Testbed via AT shell.
#
#================================================================

hexlibProjectTemplate(testbedAtCommon STATIC "hexlibShellCommon;hexlibShellCommonGpu" TRUE)

###

if (NOT DEFINED HEXLIB_AT_INC)
    set(HEXLIB_AT_INC $ENV{HEXLIB_AT_INC})
endif()

target_include_directories(testbedAtCommon PUBLIC ${HEXLIB_AT_INC})

target_link_libraries(testbedAtCommon PUBLIC vfw32.lib)
