#!/bin/bash
# ============================================================================
# InSpectra Analytics Platform - Unix/Linux/macOS Launcher Script
# ============================================================================
# This script executes the Python launcher located in the 'core' directory.
# It ensures that the Python script is run from the project's root directory.
# ============================================================================

echo "Initializing InSpectra Analytics Platform Launcher..."

# Get the directory where this script is located to determine the project root.
# This makes the script runnable from any location.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path to the Python launcher script.
LAUNCHER_PATH="$SCRIPT_DIR/core/launcher.py"

# Check if the launcher file exists.
if [ ! -f "$LAUNCHER_PATH" ]; then
    echo ""
    echo "ERROR: Launcher script not found!"
    echo "Expected to find it at: $LAUNCHER_PATH"
    echo "Please ensure you are running this from the project's root directory."
    exit 1
fi

# Execute the Python launcher.
# We use 'python3' as is standard for modern systems.
# Changing to the script's directory ensures that relative paths inside
# the python script (like for 'requirements.txt') work correctly.
cd "$SCRIPT_DIR" || exit
python3 "$LAUNCHER_PATH" "$@"

echo ""
echo "Launcher has finished."
