# InSpectra Analytics Platform - Launcher Guide

This document explains how to use the various launcher options for the InSpectra Analytics Platform.

## 🚀 Quick Start

### Option 1: Cross-Platform Python Launcher (Recommended)
```bash
python run_launcher.py
```

### Option 2: Platform-Specific Launchers

#### Windows
```cmd
run_launcher.bat
```
*Or double-click the `run_launcher.bat` file*

#### Linux/macOS/Unix
```bash
chmod +x run_launcher.sh
./run_launcher.sh
```

### Option 3: Direct GUI Launcher
```bash
python launcher.py
```

## 📁 Launcher Files Overview

| File | Platform | Description |
|------|----------|-------------|
| `run_launcher.py` | Cross-platform | Smart launcher that detects OS and tries multiple methods |
| `run_launcher.bat` | Windows | Windows batch file with comprehensive error handling |
| `run_launcher.sh` | Linux/macOS/Unix | Shell script with dependency checking |
| `launcher.py` | Cross-platform | GUI launcher with environment analysis |

## 🔧 Features

### GUI Launcher (`launcher.py`)
- **System Analysis**: Detailed system information display
- **Dependency Checking**: Checks all required Python packages
- **Automatic Installation**: One-click installation of missing packages
- **Progress Tracking**: Real-time installation progress
- **Error Reporting**: Detailed error messages and troubleshooting tips
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Platform-Specific Launchers

#### Windows Batch Launcher (`run_launcher.bat`)
- ✅ Checks for Python installation
- ✅ Verifies pip availability
- ✅ Sets proper Windows environment variables
- ✅ Creates necessary directories
- ✅ Provides detailed error messages
- ✅ Handles Windows-specific paths and permissions

#### Unix Shell Launcher (`run_launcher.sh`)
- ✅ Detects best Python version (3.8+)
- ✅ Checks for tkinter availability
- ✅ Provides distribution-specific installation commands
- ✅ Handles file permissions automatically
- ✅ Colored output for better readability
- ✅ Cross-platform Unix compatibility (Linux, macOS, BSD)

## 🐍 Python Version Requirements

- **Minimum**: Python 3.8
- **Recommended**: Python 3.9 or higher
- **Required modules**: tkinter (usually included with Python)

## 📦 Dependencies

### Core Requirements
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.1.0
- seaborn >= 0.11.0
- duckdb >= 0.8.0

### Optional but Recommended
- psutil (for system memory information)
- polars (for faster data processing)
- plotly (for interactive visualizations)

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. "Python not found" Error
**Windows:**
```cmd
# Install Python from python.org and check "Add to PATH"
# Or use Microsoft Store version
winget install Python.Python.3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-tk
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install python3 python3-pip python3-tkinter
```

**macOS:**
```bash
# Using Homebrew
brew install python-tk

# Or using official installer from python.org
```

#### 2. "tkinter not available" Error
**Linux:**
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/RHEL/Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

**macOS:**
```bash
# Usually included with Python, but if missing:
brew install python-tk
```

#### 3. "pip not available" Error
```bash
# Download and install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### 4. Permission Errors (Linux/macOS)
```bash
# Make launcher executable
chmod +x run_launcher.sh

# Or run with bash explicitly
bash run_launcher.sh
```

#### 5. Missing Dependencies
The GUI launcher will automatically offer to install missing packages. You can also install manually:
```bash
pip install -r requirements.txt
```

## 🖥️ Platform-Specific Notes

### Windows
- Uses `.bat` file for native Windows experience
- Automatically sets UTF-8 encoding
- Creates logs directory in current folder
- Handles Windows path separators correctly

### Linux
- Detects distribution and provides specific installation commands
- Checks for tkinter availability (common issue on Linux)
- Handles different Python executable names
- Provides colored terminal output

### macOS
- Compatible with both system Python and Homebrew Python
- Handles macOS-specific tkinter installation
- Works with both Intel and Apple Silicon Macs

## 🔄 Environment Variables

The launchers set these environment variables for optimal performance:

- `PYTHONPATH`: Includes current directory
- `PYTHONIOENCODING`: Set to utf-8
- `PYTHONUNBUFFERED`: Set to 1 for real-time output

## 📊 Launcher Selection Guide

| Use Case | Recommended Launcher |
|----------|---------------------|
| Windows desktop shortcut | `run_launcher.bat` |
| Linux/macOS terminal | `run_launcher.sh` |

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look in the `logs/` directory for detailed error information
2. **Try different launchers**: If one fails, try another method
3. **Update Python**: Ensure you have Python 3.8 or higher
4. **Check permissions**: Make sure you have write access to the directory
5. **Install dependencies manually**: Run `pip install -r requirements.txt`

For more help, check the main README.md or create an issue in the project repository.

## 🔧 Advanced Usage

### Console Mode
Force console mode without GUI:
```bash
python run_launcher.py --console
```

### Silent Installation
For automated setups, you can pre-install dependencies:
```bash
pip install -r requirements.txt
python main_gui.py
```

### Custom Python Installation
If you have multiple Python versions:
```bash
# Use specific Python version
python3.11 launcher.py

# Or with full path
/usr/bin/python3 launcher.py
```

---

**Note**: The GUI launcher (`launcher.py`) provides the most comprehensive experience with system analysis, dependency management, and troubleshooting guidance. Use it when setting up for the first time or when encountering issues.