@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 29528)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 17928)

del /F cleanup-ansys-EZBook-17928.bat
