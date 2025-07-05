#!/usr/bin/env python3
"""
Inspectra - Advanced Forensic Analysis Platform

Advanced interface focusing on:
- Dataset display and filtering
- Conditional analysis options
- Interactive GUI without redundancy
"""

import sys
import logging
import traceback
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.qt_simple import QApplication, QMessageBox, QStyleFactory
from utils.helpers import setup_logging
from gui.main_window import MainWindow
from config import config


def setup_application():
    """Setup the QApplication with proper configuration."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Inspectra")
    app.setOrganizationDomain("inspectra.local")

    # Apply dark theme
    apply_dark_theme(app)

    return app


def apply_dark_theme(app):
    """Apply dark theme to the application."""
    try:
        # Set fusion style for better dark theme support
        app.setStyle(QStyleFactory.create('Fusion'))

        # Dark theme stylesheet
        dark_stylesheet = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            selection-background-color: #0078d4;
        }

        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }

        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #106ebe;
        }

        QPushButton:pressed {
            background-color: #005a9e;
        }

        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }

        QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
            background-color: #404040;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
            color: #ffffff;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border: 2px solid #0078d4;
        }

        QComboBox::drop-down {
            border: none;
            background-color: #555555;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
        }

        QTableWidget {
            background-color: #404040;
            alternate-background-color: #4a4a4a;
            gridline-color: #555555;
            selection-background-color: #0078d4;
        }

        QTableWidget::item {
            padding: 5px;
        }

        QHeaderView::section {
            background-color: #555555;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #666666;
            font-weight: bold;
        }

        QListWidget {
            background-color: #404040;
            border: 1px solid #555555;
            selection-background-color: #0078d4;
        }

        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #555555;
        }

        QListWidget::item:hover {
            background-color: #4a4a4a;
        }

        QScrollBar:vertical {
            background-color: #404040;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #666666;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #777777;
        }

        QScrollBar:horizontal {
            background-color: #404040;
            height: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal {
            background-color: #666666;
            border-radius: 6px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #777777;
        }

        QMenuBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-bottom: 1px solid #555555;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 5px 10px;
        }

        QMenuBar::item:selected {
            background-color: #0078d4;
        }

        QMenu {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }

        QMenu::item {
            padding: 5px 20px;
        }

        QMenu::item:selected {
            background-color: #0078d4;
        }

        QStatusBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-top: 1px solid #555555;
        }

        QProgressBar {
            border: 1px solid #555555;
            border-radius: 3px;
            text-align: center;
            background-color: #404040;
        }

        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }

        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
            background-color: #404040;
            border: 1px solid #555555;
            border-radius: 3px;
        }

        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background-color: #0078d4;
        }

        QRadioButton::indicator {
            border-radius: 8px;
        }

        QLabel {
            color: #ffffff;
        }
        """

        app.setStyleSheet(dark_stylesheet)

    except Exception as e:
        logging.warning(f"Failed to apply dark theme: {e}")


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    # Check critical dependencies
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")

    if missing_deps:
        error_msg = f"""
        Missing required dependencies: {', '.join(missing_deps)}

        Please install them using:
        pip install {' '.join(missing_deps)}
        """

        QMessageBox.critical(None, "Missing Dependencies", error_msg)
        return False

    return True


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception
    logger = logging.getLogger(__name__)
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Show error dialog
    error_msg = f"""
    An unexpected error occurred:

    {exc_type.__name__}: {exc_value}

    Please check the log file for more details.
    """

    QMessageBox.critical(None, "Unexpected Error", error_msg)


def main():
    """Main application entry point."""
    try:
        # Setup logging first
        logger = setup_logging("INFO")
        logger.info("Starting Inspectra")

        # Setup global exception handler
        sys.excepthook = handle_exception

        # Create application
        app = setup_application()

        # Check dependencies
        if not check_dependencies():
            return 1

        # Create and show main window
        try:
            main_window = MainWindow()
            main_window.show()

            logger.info("Application started successfully")

            # Run application event loop
            return app.exec_()

        except Exception as e:
            logger.critical(f"Failed to create main window: {e}")
            QMessageBox.critical(None, "Startup Error",
                                 f"Failed to start application:\n{str(e)}")
            return 1

    except Exception as e:
        # Fallback error handling if logging setup fails
        print(f"Critical startup error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())