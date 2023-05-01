@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 30480)
if /i "%LOCALHOST%"=="EZBook" (taskkill /f /pid 34052)

del /F cleanup-ansys-EZBook-34052.bat
