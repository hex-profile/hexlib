@echo off

rem ================================================================
rem 
rem Prepare.
rem 
rem ================================================================

set REQUIRE=exit /b 1

set sourceDir=%~1
set resultDir=%~2
set generatorName=%~3

if [%sourceDir%] == [] goto error
if [%resultDir%] == [] goto error
if [%generatorName%] == [] goto error

rem ================================================================
rem 
rem  Make GPU compiler.
rem 
rem ================================================================

echo Building GPU compiler...

call "setup_cl_x64.cmd" || %REQUIRE%

mkdir "%resultDir%" 2>nul
cd /D "%resultDir%" || %REQUIRE%

cmake ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%resultDir% ^
    -G"%generatorName%" "%sourceDir%" ^
    -DHEXLIB_PLATFORM=0 ^
    -DHEXLIB_GUARDED_MEMORY=0 ^
    || %REQUIRE%

cmake --build . --target gpuCompiler -- || %REQUIRE%

rem ----------------------------------------------------------------

goto exit

:error

echo Arguments are required: sourceDir resultDir generatorName

:exit
