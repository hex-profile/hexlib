#================================================================
#
# Testbed via AT shell.
#
#================================================================

hexlibProjectTemplate(testbedAt STATIC . "moduleLibGpu;shellCommon;shellCommonGpu" TRUE hexlib)

###

if (NOT DEFINED HEXLIB_AT_INC)
    set(HEXLIB_AT_INC $ENV{HEXLIB_AT_INC})
endif()

target_include_directories(testbedAt PUBLIC ${HEXLIB_AT_INC})

target_link_libraries(testbedAt PUBLIC vfw32.lib)
