@echo off
REM ============================================================================
REM InSpectra Analytics Platform - Windows Launcher Script
REM ============================================================================
REM This script executes the Python launcher located in the 'core' directory.
REM It ensures that the Python script is run from the project's root directory.
REM ============================================================================

echo Initializing InSpectra Analytics Platform Launcher...

REM Get the directory where this batch file is located. This is the project root.
set "SCRIPT_DIR=%~dp0"

REM Define the path to the Python launcher script.
set "LAUNCHER_PATH=%SCRIPT_DIR%core\launcher.py"

REM Check if the launcher file exists.
if not exist "%LAUNCHER_PATH%" (
    echo.
    echo ERROR: Launcher script not found!
    echo Expected to find it at: %LAUNCHER_PATH%
    echo Please ensure you are running this from the project's root directory.
    pause
    exit /b 1
)

REM Execute the Python launcher.
REM We use 'python' which should be in the system's PATH.
REM Using "%SCRIPT_DIR%" as the working directory ensures that relative paths
REM inside the python script (like for 'requirements.txt') work correctly.
pushd "%SCRIPT_DIR%"
python "%LAUNCHER_PATH%" %*
popd

echo.
echo Launcher has finished.
