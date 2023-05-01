@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 7108)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 19320)

del /F cleanup-ansys-EZBook-19320.bat
