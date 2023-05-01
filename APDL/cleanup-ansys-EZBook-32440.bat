@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 33056)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 32440)

del /F cleanup-ansys-EZBook-32440.bat
