@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 6568)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 20404)

del /F cleanup-ansys-EZBook-20404.bat
