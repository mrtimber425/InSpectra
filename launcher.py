import subprocess
import sys
import os
import importlib
import venv
from pathlib import Path
import platform
import shutil

# --- Dependency Installation Logic (adapted from install_dependencies.py) ---

def check_package(package_name, import_name=None, python_executable=sys.executable):
    """
    Check if a package is installed and importable using a specific Python executable.
    This runs a subprocess to check the target environment.
    """
    if import_name is None:
        import_name = package_name

    try:
        # Run a subprocess to attempt importing the module in the target Python environment
        subprocess.run(
            [python_executable, "-c", f"import {import_name}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10 # Add a timeout to prevent hanging
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking {package_name}: {e}")
        return False

def install_package(package_name, description="", python_executable=sys.executable):
    """
    Install a package using pip for a specific Python executable.
    """
    print(f"Installing {package_name}... {description}")
    try:
        # Use the specified python_executable to run pip
        subprocess.check_call(
            [python_executable, "-m", "pip", "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e.output}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during installation of {package_name}: {e}")
        return False

def run_dependency_installer(python_executable):
    """
    Installs core and enhanced dependencies using the specified Python executable.
    This function incorporates the main logic of the original install_dependencies.py.
    """
    print("=" * 60)
    print("Inspectra - Dependency Installer")
    print("=" * 60)

    # Core dependencies (required for basic functionality)
    core_deps = [
        ("PyQt5", "PyQt5", "GUI framework"),
        ("pandas", "pandas", "Data processing"),
        ("numpy", "numpy", "Numerical computing"),
        ("matplotlib", "matplotlib", "Basic plotting"),
        ("seaborn", "seaborn", "Statistical visualization"),
        ("scikit-learn", "sklearn", "Machine learning"),
        ("openpyxl", "openpyxl", "Excel file support")
    ]

    # Enhanced dependencies (optional but recommended for full features)
    enhanced_deps = [
        ("xgboost", "xgboost", "Advanced ML algorithms"),
        ("imbalanced-learn", "imblearn", "Handling imbalanced datasets"),
        ("shap", "shap", "ML model interpretability"),
        ("plotly", "plotly", "Interactive visualizations"),
        ("dash", "dash", "Web-based dashboards"),
        ("bokeh", "bokeh", "Interactive plotting"),
        ("altair", "altair", "Statistical visualization"),
        ("fpdf2", "fpdf", "PDF generation"),
        ("reportlab", "reportlab", "Advanced PDF reports"),
        ("xlsxwriter", "xlsxwriter", "Excel writing"),
        ("python-dateutil", "dateutil", "Date parsing"),
        ("pytz", "pytz", "Timezone handling"),
        ("requests", "requests", "HTTP requests"),
        ("Pillow", "PIL", "Image processing")
    ]

    print("\nChecking core dependencies...")
    print("-" * 40)

    # Identify missing core dependencies
    missing_core = []
    for package, import_name, description in core_deps:
        if not check_package(package, import_name, python_executable):
            print(f"✗ {package} - {description} (MISSING)")
            missing_core.append((package, description))
        else:
            print(f"✓ {package} - {description}")

    print("\nChecking enhanced dependencies...")
    print("-" * 40)

    # Identify missing enhanced dependencies
    missing_enhanced = []
    for package, import_name, description in enhanced_deps:
        if not check_package(package, import_name, python_executable):
            print(f"✗ {package} - {description} (MISSING)")
            missing_enhanced.append((package, description))
        else:
            print(f"✓ {package} - {description}")

    # Install missing core dependencies
    if missing_core:
        print(f"\n🚨 CRITICAL: {len(missing_core)} core dependencies missing!")
        print("Attempting to install core dependencies...")
        print("-" * 40)
        for package, description in missing_core:
            install_package(package, description, python_executable)
    else:
        print("\nAll core dependencies already installed.")

    # Install missing enhanced dependencies (non-interactively for automation)
    if missing_enhanced:
        print(f"\n⚠️  {len(missing_enhanced)} enhanced dependencies missing.")
        print("Attempting to install enhanced dependencies for full functionality...")
        print("-" * 40)
        for package, description in missing_enhanced:
            install_package(package, description, python_executable)
    else:
        print("\nAll enhanced dependencies already installed.")

    # Final check after installation attempts
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)

    all_core_working = True
    print("\nCore Dependencies:")
    for package, import_name, description in core_deps:
        if check_package(package, import_name, python_executable):
            print(f"✓ {package}")
        else:
            print(f"✗ {package} (STILL MISSING)")
            all_core_working = False

    print("\nEnhanced Dependencies:")
    available_enhanced = 0
    for package, import_name, description in enhanced_deps:
        if check_package(package, import_name, python_executable):
            print(f"✓ {package}")
            available_enhanced += 1
        else:
            print(f"- {package} (not installed)")

    print(f"\nEnhanced features available: {available_enhanced}/{len(enhanced_deps)} ({available_enhanced/len(enhanced_deps)*100:.0f}%)")

    if all_core_working:
        print("\n🎉 All core dependencies installed successfully!")
        print("The application should now run without import errors.")
        return True
    else:
        print("\n❌ Some core dependencies are still missing.")
        print("Please review the installation logs and try to install them manually.")
        return False

# --- Launcher Specific Functions ---

VENV_NAME = "inspectra_env"
MAIN_APP_SCRIPT = "main.py"

def find_system_python_executable():
    """
    Finds a suitable Python 3 executable on the system (Python 3.8 or higher).
    Prioritizes 'python3' then 'python'.
    """
    python_executables = ["python3", "python"]
    for py_exe in python_executables:
        try:
            # Check if it's Python 3 and at least 3.8
            result = subprocess.run(
                [py_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, check=True, timeout=5
            )
            version_str = result.stdout.strip()
            major, minor = map(int, version_str.split('.'))
            if major == 3 and minor >= 8:
                print(f"Found suitable Python: {py_exe} (Version {version_str})")
                return py_exe
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            continue
    return None

def create_or_get_venv_python(base_python_exe):
    """
    Creates a virtual environment if it doesn't exist, or returns the path
    to its Python executable if it does.
    """
    venv_dir = Path(VENV_NAME)
    if platform.system() == "Windows":
        venv_python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python_exe = venv_dir / "bin" / "python"

    if not venv_dir.exists():
        print(f"\nCreating virtual environment '{VENV_NAME}'...")
        try:
            # Use the base_python_exe to create the venv
            subprocess.check_call([base_python_exe, "-m", "venv", str(venv_dir)])
            print(f"Virtual environment '{VENV_NAME}' created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            print("Please ensure you have the 'venv' module installed (e.g., `sudo apt install python3-venv` on Linux).")
            return None
    else:
        print(f"\nVirtual environment '{VENV_NAME}' already exists.")

    if not venv_python_exe.exists():
        print(f"Error: Python executable not found in '{venv_python_exe}'. Virtual environment might be corrupted.")
        print(f"Attempting to remove and recreate '{VENV_NAME}'...")
        try:
            shutil.rmtree(venv_dir)
            subprocess.check_call([base_python_exe, "-m", "venv", str(venv_dir)])
            print(f"Virtual environment '{VENV_NAME}' recreated successfully.")
        except Exception as e:
            print(f"Failed to recreate virtual environment: {e}")
            return None

    return str(venv_python_exe)

def main():
    """
    Main function for the Inspectra launcher.
    Orchestrates environment setup, dependency installation, and application launch.
    """
    print("=" * 60)
    print("Inspectra - Automated Setup and Launcher")
    print("=" * 60)
    print("This script will set up a virtual environment, install dependencies, and launch Inspectra.")
    print("Please be patient, this may take a few minutes.")

    # 1. Find a suitable base Python executable on the system
    print("\n" + "=" * 60)
    print("STEP 1: Checking Python Environment")
    print("=" * 60)
    base_python = find_system_python_executable()
    if not base_python:
        print("\nERROR: Python 3.8 or higher not found on your system or not accessible from PATH.")
        print("Please install Python 3.8+ and ensure it's in your system's PATH.")
        print("Download from: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"Using system Python: {base_python}")

    # 2. Create or get the virtual environment's Python executable
    venv_python = create_or_get_venv_python(base_python)
    if not venv_python:
        print("\nERROR: Could not set up the virtual environment.")
        sys.exit(1)
    print(f"Using virtual environment Python: {venv_python}")

    # 3. Install dependencies within the virtual environment
    print("\n" + "=" * 60)
    print("STEP 2: Installing/Checking Dependencies")
    print("=" * 60)
    dependencies_installed_successfully = run_dependency_installer(venv_python)

    if not dependencies_installed_successfully:
        print("\nCRITICAL ERROR: Not all core dependencies could be installed.")
        print("Please review the errors above and try to install them manually.")
        print(f"You can try running '{venv_python} -m pip install -r requirements.txt' inside your terminal after navigating to this directory.")
        sys.exit(1)

    # 4. Launch the main application
    print("\n" + "=" * 60)
    print("STEP 3: Launching Inspectra")
    print("=" * 60)
    main_script_path = Path(MAIN_APP_SCRIPT)
    if not main_script_path.exists():
        print(f"ERROR: Main application script '{MAIN_APP_SCRIPT}' not found in the current directory.")
        print("Please ensure 'main.py' is in the same directory as this launcher script.")
        sys.exit(1)

    try:
        print(f"Running: {venv_python} {MAIN_APP_SCRIPT}")
        # Execute main.py using the virtual environment's python
        # We use Popen to allow the main application to run independently
        # and not block the launcher, but still keep the console open.
        process = subprocess.Popen([venv_python, str(main_script_path)])
        process.wait() # Wait for the application to close
        print(f"\nInspectra application exited with code {process.returncode}.")
    except Exception as e:
        print(f"An unexpected error occurred while launching Inspectra: {e}")
        sys.exit(1)

    print("\nSetup and launch process completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()