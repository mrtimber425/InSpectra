@echo off
title Inspectra Launcher

echo.
echo ============================================================
echo  Inspectra - Automated Setup and Launcher (Windows)
echo ============================================================
echo.

REM --- Find Python ---
REM Check if 'python' is directly in PATH and is Python 3
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import sys; exit(sys.version_info.major != 3)" >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_EXE=python
        goto :run_launcher
    )
)

REM Check if 'py' launcher is available (Windows specific)
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3 -c "import sys; exit(sys.version_info.major != 3)" >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_EXE=py -3
        goto :run_launcher
    )
)

echo ERROR: Python 3.8 or higher not found or not accessible from PATH.
echo Please install Python 3.8+ and ensure it's added to your system's PATH.
echo Download from: https://www.python.org/downloads/
echo.
pause
exit /b 1

:run_launcher
echo Running Python launcher script...
%PYTHON_EXE% launcher.py
if %errorlevel% neq 0 (
    echo.
    echo An error occurred during setup or launch.
    echo Please check the messages above for details.
)
echo.
echo Press any key to close this window...
pause
exit /b %errorlevel%
