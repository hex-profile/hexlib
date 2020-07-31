@echo off

rem ================================================================
rem 
rem Prepare.
rem 
rem ================================================================

set REQUIRE=exit /b 1

set sourceDir=%~1
set intermDir=%~2
set resultDir=%~3
set generatorName=%~4

if [%sourceDir%] == [] goto error
if [%intermDir%] == [] goto error
if [%resultDir%] == [] goto error
if [%generatorName%] == [] goto error

rem ================================================================
rem 
rem  Make GPU compiler.
rem 
rem ================================================================

rem ----------------------------------------------------------------

echo Building GPU compiler...

call "setup_cl_x64.cmd" || %REQUIRE%

mkdir "%resultDir%" 2>nul
mkdir "%intermDir%" 2>nul
cd /D "%intermDir%" || %REQUIRE%

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

echo Arguments are required: sourceDir intermDir resultDir generatorName

:exit
