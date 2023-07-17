@echo off
set REQUIRE=exit /b 1

diff -r archive %HEXLIB_OUTPUT%\archive
if %ERRORLEVEL% equ 0 echo No differences found & exit /b 0

echo Archiving...

del archive.7z
7z a -p%HEXLIB_ARCHIVE_PASSWORD% -mhe=on -mtc=off archive archive || %REQUIRE%

rmdir /q /s "%HEXLIB_OUTPUT%\archive"
xcopy /q /e archive "%HEXLIB_OUTPUT%\archive\"
