#!/usr/bin/env python3
"""
CyberForensics Data Detective - Dependency Installation Script
Automatically installs required and optional dependencies
"""

import subprocess
import sys
import os
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name, description=""):
    """Install a package using pip"""
    print(f"Installing {package_name}... {description}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("CyberForensics Data Detective - Dependency Installer")
    print("=" * 60)
    
    # Core dependencies (required)
    core_deps = [
        ("PyQt5", "PyQt5", "GUI framework"),
        ("pandas", "pandas", "Data processing"),
        ("numpy", "numpy", "Numerical computing"),
        ("matplotlib", "matplotlib", "Basic plotting"),
        ("seaborn", "seaborn", "Statistical visualization"),
        ("scikit-learn", "sklearn", "Machine learning"),
        ("openpyxl", "openpyxl", "Excel file support")
    ]
    
    # Enhanced dependencies (optional but recommended)
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
    
    print("\\nChecking core dependencies...")
    print("-" * 40)
    
    missing_core = []
    for package, import_name, description in core_deps:
        if check_package(package, import_name):
            print(f"✓ {package} - {description}")
        else:
            print(f"✗ {package} - {description} (MISSING)")
            missing_core.append((package, description))
    
    print("\\nChecking enhanced dependencies...")
    print("-" * 40)
    
    missing_enhanced = []
    for package, import_name, description in enhanced_deps:
        if check_package(package, import_name):
            print(f"✓ {package} - {description}")
        else:
            print(f"✗ {package} - {description} (MISSING)")
            missing_enhanced.append((package, description))
    
    # Install missing dependencies
    if missing_core:
        print(f"\\n🚨 CRITICAL: {len(missing_core)} core dependencies missing!")
        print("Installing core dependencies...")
        print("-" * 40)
        
        for package, description in missing_core:
            install_package(package, description)
    
    if missing_enhanced:
        print(f"\\n⚠️  {len(missing_enhanced)} enhanced dependencies missing.")
        
        response = input("\\nInstall enhanced dependencies for full functionality? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("Installing enhanced dependencies...")
            print("-" * 40)
            
            for package, description in missing_enhanced:
                install_package(package, description)
        else:
            print("\\nSkipping enhanced dependencies.")
            print("Note: Some advanced features may not be available.")
    
    # Final check
    print("\\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    
    all_working = True
    
    print("\\nCore Dependencies:")
    for package, import_name, description in core_deps:
        if check_package(package, import_name):
            print(f"✓ {package}")
        else:
            print(f"✗ {package} (STILL MISSING)")
            all_working = False
    
    print("\\nEnhanced Dependencies:")
    available_enhanced = 0
    for package, import_name, description in enhanced_deps:
        if check_package(package, import_name):
            print(f"✓ {package}")
            available_enhanced += 1
        else:
            print(f"- {package} (not installed)")
    
    print(f"\\nEnhanced features available: {available_enhanced}/{len(enhanced_deps)} ({available_enhanced/len(enhanced_deps)*100:.0f}%)")
    
    if all_working:
        print("\\n🎉 All core dependencies installed successfully!")
        print("The application should now run without import errors.")
    else:
        print("\\n❌ Some core dependencies are still missing.")
        print("Please install them manually or check for installation errors.")
    
    print("\\nTo run the application:")
    print("  python main.py")
    
    print("\\nFor help with installation issues:")
    print("  - Try: pip install --upgrade pip")
    print("  - Try: pip install --user <package_name>")
    print("  - Check Python version compatibility")

if __name__ == "__main__":
    main()

