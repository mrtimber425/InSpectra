# InSpectra Data Processing Platform - Launcher Guide

InSpectra is a flexible data processing platform that empowers users to upload and run their own AI models to perform analytics. This guide outlines how to launch the InSpectra platform across different environments.

---

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

---

## 📁 Launcher Files Overview

| File               | Platform         | Description                                                          |
| ------------------ | ---------------- | -------------------------------------------------------------------- |
| `run_launcher.py`  | Cross-platform   | Intelligent launcher that detects OS and initializes the environment |
| `run_launcher.bat` | Windows          | Batch file for seamless setup and execution on Windows               |
| `run_launcher.sh`  | Linux/macOS/Unix | Shell script with setup automation for Unix-based systems            |
| `launcher.py`      | Cross-platform   | GUI launcher with environment diagnostics and model loading support  |

---

## 🧠 What is InSpectra?

InSpectra is a data processing and AI analytics tool designed for users who want to:

* **Upload their own AI models**
* **Process and analyze data efficiently**
* **Visualize analytics using interactive tools**
* **Perform model-specific or custom data pipelines**

Whether you're working with machine learning, statistical models, or data cleaning workflows, InSpectra provides a flexible launchpad.

---

## 🔧 Features

### GUI Launcher (`launcher.py`)

* **Model Uploading**: Load and run your own AI/ML models
* **System Diagnostics**: Get detailed environment checks
* **Dependency Management**: Auto-checks and installs required Python packages
* **Progress Feedback**: Real-time installation and setup status
* **Error Reporting**: User-friendly error guidance
* **Cross-Platform Support**: Windows, macOS, Linux

### Platform-Specific Launchers

#### Windows (`run_launcher.bat`)

* Verifies Python and pip installations
* Configures Windows-specific environment variables
* Handles permissions and directories
* Displays descriptive error messages

#### Unix Shell (`run_launcher.sh`)

* Detects and prefers Python 3.8+
* Ensures tkinter and pip availability
* Provides distro-specific instructions
* Uses color-coded terminal messages

---

## 🐍 Python Version Requirements

* **Minimum**: Python 3.8
* **Recommended**: Python 3.9+
* **Must Include**: `tkinter` for GUI support

---

## 📦 Core Dependencies

### Required Packages

* `pandas >= 1.5.0`
* `numpy >= 1.21.0`
* `matplotlib >= 3.5.0`
* `scikit-learn >= 1.1.0`
* `seaborn >= 0.11.0`
* `duckdb >= 0.8.0`

### Optional Enhancements

* `psutil` – for memory diagnostics
* `polars` – high-speed data processing
* `plotly` – interactive data visualization

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Python Not Found**

**Windows:**

```cmd
winget install Python.Python.3.11
```

**Linux/macOS:** Install using your package manager (e.g., `apt`, `dnf`, or `brew`).

#### 2. **tkinter Missing**

Install `python3-tk` or `python-tkinter` based on your OS.

#### 3. **pip Not Installed**

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### 4. **Permissions**

```bash
chmod +x run_launcher.sh
bash run_launcher.sh
```

#### 5. **Dependencies Missing**

Install via:

```bash
pip install -r requirements.txt
```

---

## 🖥️ OS-Specific Notes

### Windows

* Native `.bat` launcher
* Automatically configures paths and encoding
* Handles directory creation

### Linux/macOS

* Detects Python executable
* Provides actionable tkinter installation steps
* Supports both Intel and ARM-based Macs

---

## 🔄 Environment Variables Set by Launchers

* `PYTHONPATH`: Includes current working directory
* `PYTHONIOENCODING`: Set to `utf-8`
* `PYTHONUNBUFFERED`: Enables real-time output

---

## 📊 Launcher Selection Guide

| Use Case                               | Recommended Launcher |
| -------------------------------------- | -------------------- |
| General Use (Any OS)                   | `run_launcher.py`    |
| Windows users                          | `run_launcher.bat`   |
| Linux/macOS users                      | `run_launcher.sh`    |
| GUI with model upload and system check | `launcher.py`        |

---

## 🆘 Getting Help

* Check logs in the `logs/` directory for errors
* Try alternative launchers if one fails
* Manually install dependencies via `pip install -r requirements.txt`
* Ensure Python 3.8+ is installed with `tkinter`

For more help, see the main `README.md` or open an issue on the repository.

---

## 🔧 Advanced Usage

### Console-Only Mode

```bash
python run_launcher.py --console
```

### Silent Installation

```bash
pip install -r requirements.txt
python main_gui.py
```

### Custom Python Executable

```bash
python3.11 launcher.py
# or
/usr/bin/python3 launcher.py
```

---

**Note**: Use the GUI launcher (`launcher.py`) for the full experience: model uploading, system compatibility checks, and guided setup.

---
