#!/bin/bash

echo ""
echo "============================================================"
echo " Inspectra - Automated Setup and Launcher (Linux/macOS)"
echo "============================================================"
echo ""

# --- Find Python ---
# Check if 'python3' command exists and is Python 3
PYTHON_EXE=""
if command -v python3 &> /dev/null; then
    if python3 -c "import sys; exit(sys.version_info.major != 3)" &> /dev/null; then
        PYTHON_EXE="python3"
    fi
fi

# Fallback to 'python' if 'python3' is not found or not Python 3, and 'python' is Python 3
if [ -z "$PYTHON_EXE" ] && command -v python &> /dev/null; then
    if python -c "import sys; exit(sys.version_info.major != 3)" &> /dev/null; then
        PYTHON_EXE="python"
    fi
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "ERROR: Python 3.8 or higher not found on your system."
    echo "Please install Python 3.8+."
    echo "For Debian/Ubuntu: sudo apt install python3 python3-venv"
    echo "For Fedora: sudo dnf install python3 python3-virtualenv"
    echo "For macOS (using Homebrew): brew install python@3.x"
    echo ""
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

echo "Using Python executable: $PYTHON_EXE"
echo "Running Python launcher script..."
"$PYTHON_EXE" launcher.py
LAUNCHER_EXIT_CODE=$?

if [ $LAUNCHER_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "An error occurred during setup or launch."
    echo "Please check the messages above for details."
fi

echo ""
read -n 1 -s -r -p "Press any key to close this window..."
exit $LAUNCHER_EXIT_CODE
