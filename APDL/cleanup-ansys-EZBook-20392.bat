@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 29372)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 20392)

del /F cleanup-ansys-EZBook-20392.bat
