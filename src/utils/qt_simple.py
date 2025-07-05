"""
Simplified PyQt imports for CyberForensics Data Detective.
No complex compatibility layer - just direct imports.
"""

import sys

# Try PyQt5 first (more stable and widely available)
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    QT_VERSION = 5
    print("Using PyQt5")
    
    # PyQt5 specific signal
    Signal = pyqtSignal
    
except ImportError:
    try:
        from PyQt6.QtWidgets import *
        from PyQt6.QtCore import *
        from PyQt6.QtGui import *
        QT_VERSION = 6
        print("Using PyQt6")
        
        # PyQt6 uses Signal instead of pyqtSignal
        try:
            Signal = pyqtSignal
        except NameError:
            pass
            
    except ImportError:
        print("ERROR: Neither PyQt5 nor PyQt6 is installed!")
        print("Please install PyQt5: pip install PyQt5")
        print("Or PyQt6: pip install PyQt6")
        sys.exit(1)

# Common application setup
def setup_app():
    """Setup QApplication with proper settings."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    app.setApplicationName("CyberForensics Data Detective")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("CyberForensics Team")
    
    # Enable high DPI scaling
    if QT_VERSION == 5:
        try:
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        except AttributeError:
            pass
    
    return app

