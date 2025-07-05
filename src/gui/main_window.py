"""
Inspectra Main Window with Context-Aware Visualizations
Analysis results display context-specific visualizations based on forensic analysis type
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from datetime import datetime
from collections import Counter

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from core.data_processor import DataProcessor
from core.export_manager import ExportManager
from detection.network_anomaly import NetworkAnomalyDetector
from detection.financial_fraud import FinancialFraudDetector
from utils.advanced_visualizations import AdvancedVisualizationWidget, VisualizationManager

class ProfessionalTheme:
    """Professional corporate theme colors and styles."""

    # Main colors
    BACKGROUND = "#1a1a1a"              # Main background
    PANEL = "#2d2d2d"                   # Panel backgrounds
    HEADER = "#333333"                  # Header backgrounds
    SURFACE = "#404040"                 # Surface elements
    TAB_ACTIVE = "#0078d4"              # Active tab color
    TAB_INACTIVE = "#555555"            # Inactive tab color

    # Text colors
    TEXT_PRIMARY = "#ffffff"            # Primary text
    TEXT_SECONDARY = "#cccccc"          # Secondary text
    TEXT_MUTED = "#999999"              # Muted text

    # Accent colors
    ACCENT_BLUE = "#0078d4"             # Primary blue
    ACCENT_BLUE_HOVER = "#106ebe"       # Blue hover
    ACCENT_GREEN = "#107c10"            # Success green
    ACCENT_ORANGE = "#ff8c00"           # Warning orange
    ACCENT_RED = "#d13438"              # Error red

    # Interactive elements
    BUTTON = "#404040"                  # Button background
    BUTTON_HOVER = "#505050"            # Button hover
    BUTTON_ACTIVE = "#606060"           # Button active

    # Borders
    BORDER = "#555555"                  # Default border
    BORDER_LIGHT = "#666666"            # Light border

    @classmethod
    def get_stylesheet(cls):
        return f"""
        /* Main Window */
        QMainWindow {{
            background-color: {cls.BACKGROUND};
            color: {cls.TEXT_PRIMARY};
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        
        /* Tab Widget */
        QTabWidget::pane {{
            border: 2px solid {cls.BORDER};
            border-radius: 8px;
            background-color: {cls.PANEL};
        }}
        
        QTabWidget::tab-bar {{
            alignment: left;
        }}
        
        QTabBar::tab {{
            background-color: {cls.TAB_INACTIVE};
            color: {cls.TEXT_SECONDARY};
            border: 2px solid {cls.BORDER};
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 8px 16px;
            margin-right: 2px;
            font-weight: 500;
            min-width: 120px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {cls.TAB_ACTIVE};
            color: white;
            border-color: {cls.TAB_ACTIVE};
            font-weight: bold;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {cls.BUTTON_HOVER};
            color: {cls.TEXT_PRIMARY};
        }}
        
        QTabBar::close-button {{
            image: none;
            background-color: {cls.ACCENT_RED};
            border-radius: 8px;
            width: 16px;
            height: 16px;
        }}
        
        QTabBar::close-button:hover {{
            background-color: #ff4444;
        }}
        
        /* Group Boxes */
        QGroupBox {{
            font-weight: bold;
            font-size: 13px;
            border: 2px solid {cls.BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            background-color: {cls.PANEL};
            color: {cls.TEXT_PRIMARY};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: {cls.TEXT_PRIMARY};
            background-color: {cls.PANEL};
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {cls.BUTTON};
            color: {cls.TEXT_PRIMARY};
            border: 1px solid {cls.BORDER};
            border-radius: 6px;
            padding: 10px 16px;
            font-size: 12px;
            font-weight: 600;
            min-height: 28px;
        }}
        
        QPushButton:hover {{
            background-color: {cls.BUTTON_HOVER};
            border-color: {cls.BORDER_LIGHT};
        }}
        
        QPushButton:pressed {{
            background-color: {cls.BUTTON_ACTIVE};
        }}
        
        QPushButton:disabled {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_MUTED};
            border-color: {cls.TEXT_MUTED};
        }}
        
        /* Primary Buttons */
        QPushButton[buttonClass="primary"] {{
            background-color: {cls.ACCENT_BLUE};
            border-color: {cls.ACCENT_BLUE};
            color: white;
            font-weight: bold;
        }}
        
        QPushButton[buttonClass="primary"]:hover {{
            background-color: {cls.ACCENT_BLUE_HOVER};
        }}
        
        /* Success Buttons */
        QPushButton[buttonClass="success"] {{
            background-color: {cls.ACCENT_GREEN};
            border-color: {cls.ACCENT_GREEN};
            color: white;
            font-weight: bold;
        }}
        
        /* Warning Buttons */
        QPushButton[buttonClass="warning"] {{
            background-color: {cls.ACCENT_ORANGE};
            border-color: {cls.ACCENT_ORANGE};
            color: white;
            font-weight: bold;
        }}
        
        /* Danger Buttons */
        QPushButton[buttonClass="danger"] {{
            background-color: {cls.ACCENT_RED};
            border-color: {cls.ACCENT_RED};
            color: white;
            font-weight: bold;
        }}
        
        /* Input Fields */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            border: 2px solid {cls.BORDER};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {cls.ACCENT_BLUE};
        }}
        
        /* Combo Boxes */
        QComboBox {{
            background-color: {cls.SURFACE};
            color: {cls.TEXT_PRIMARY};
            border: 2px solid {cls.BORDER};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            min-width: 120px;
        }}
        
        QComboBox:focus {{
            border-color: {cls.ACCENT_BLUE};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {cls.TEXT_SECONDARY};
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_PRIMARY};
            border: 2px solid {cls.BORDER};
            selection-background-color: {cls.ACCENT_BLUE};
        }}
        
        /* Radio Buttons */
        QRadioButton {{
            color: {cls.TEXT_PRIMARY};
            font-size: 12px;
            font-weight: 500;
            spacing: 10px;
        }}
        
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
        }}
        
        QRadioButton::indicator:unchecked {{
            border: 2px solid {cls.BORDER};
            border-radius: 8px;
            background-color: {cls.SURFACE};
        }}
        
        QRadioButton::indicator:checked {{
            border: 2px solid {cls.ACCENT_BLUE};
            border-radius: 8px;
            background-color: {cls.ACCENT_BLUE};
        }}
        
        /* Check Boxes */
        QCheckBox {{
            color: {cls.TEXT_PRIMARY};
            font-size: 12px;
            font-weight: 500;
            spacing: 10px;
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
        }}
        
        QCheckBox::indicator:unchecked {{
            border: 2px solid {cls.BORDER};
            border-radius: 3px;
            background-color: {cls.SURFACE};
        }}
        
        QCheckBox::indicator:checked {{
            border: 2px solid {cls.ACCENT_BLUE};
            border-radius: 3px;
            background-color: {cls.ACCENT_BLUE};
        }}
        
        /* Tables */
        QTableWidget {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_PRIMARY};
            border: 2px solid {cls.BORDER};
            border-radius: 8px;
            gridline-color: {cls.BORDER};
            selection-background-color: {cls.ACCENT_BLUE};
            font-size: 14px;
            font-family: 'Segoe UI', Arial, sans-serif;
            alternate-background-color: {cls.SURFACE};
            font-weight: 500;
            show-decoration-selected: 1;
        }}
        
        QTableWidget::item {{
            padding: 15px 10px;
            border-bottom: 1px solid {cls.BORDER};
            border-right: 1px solid {cls.BORDER};
            min-height: 30px;
            text-align: left;
        }}
        
        QTableWidget::item:selected {{
            background-color: {cls.ACCENT_BLUE};
            color: white;
            font-weight: bold;
        }}
        
        QHeaderView::section {{
            background-color: {cls.HEADER};
            color: {cls.TEXT_PRIMARY};
            padding: 16px 12px;
            border: 1px solid {cls.BORDER};
            font-weight: bold;
            font-size: 15px;
            min-height: 36px;
            text-align: center;
        }}
        
        /* Vertical Header (Row Numbers) */
        QHeaderView::section:vertical {{
            background-color: {cls.HEADER};
            color: white;
            padding: 12px 8px;
            border: 1px solid {cls.BORDER};
            font-weight: bold;
            font-size: 14px;
            min-width: 50px;
            text-align: center;
        }}
        
        /* Labels */
        QLabel {{
            color: {cls.TEXT_PRIMARY};
            font-size: 12px;
        }}
        
        QLabel[labelClass="header"] {{
            font-size: 16px;
            font-weight: bold;
            color: {cls.TEXT_PRIMARY};
        }}
        
        QLabel[labelClass="subheader"] {{
            font-size: 14px;
            font-weight: bold;
            color: {cls.ACCENT_BLUE};
        }}
        
        QLabel[labelClass="secondary"] {{
            color: {cls.TEXT_SECONDARY};
        }}
        
        QLabel[labelClass="muted"] {{
            color: {cls.TEXT_MUTED};
        }}
        
        QLabel[labelClass="success"] {{
            color: {cls.ACCENT_GREEN};
            font-weight: bold;
        }}
        
        QLabel[labelClass="warning"] {{
            color: {cls.ACCENT_ORANGE};
            font-weight: bold;
        }}
        
        QLabel[labelClass="danger"] {{
            color: {cls.ACCENT_RED};
            font-weight: bold;
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_SECONDARY};
            border-top: 2px solid {cls.BORDER};
            font-size: 11px;
            padding: 4px;
        }}
        
        /* Menu Bar */
        QMenuBar {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_PRIMARY};
            border-bottom: 2px solid {cls.BORDER};
            font-size: 12px;
            padding: 4px;
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {cls.ACCENT_BLUE};
        }}
        
        QMenu {{
            background-color: {cls.PANEL};
            color: {cls.TEXT_PRIMARY};
            border: 2px solid {cls.BORDER};
            border-radius: 6px;
        }}
        
        QMenu::item {{
            padding: 8px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {cls.ACCENT_BLUE};
        }}
        
        /* Scroll Bars */
        QScrollBar:vertical {{
            background-color: {cls.PANEL};
            width: 14px;
            border-radius: 7px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {cls.BORDER_LIGHT};
            border-radius: 7px;
            min-height: 24px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {cls.TEXT_MUTED};
        }}
        
        QScrollBar:horizontal {{
            background-color: {cls.PANEL};
            height: 14px;
            border-radius: 7px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {cls.BORDER_LIGHT};
            border-radius: 7px;
            min-width: 24px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {cls.TEXT_MUTED};
        }}
        
        /* Splitter */
        QSplitter::handle {{
            background-color: {cls.BORDER};
        }}
        
        QSplitter::handle:horizontal {{
            width: 3px;
        }}
        
        QSplitter::handle:vertical {{
            height: 3px;
        }}
        """

class SeverityCard(QWidget):
    """Professional severity indicator card."""

    def __init__(self, title, color, count=0):
        super().__init__()
        self.title = title
        self.color = color
        self.count = count
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Title
        title_label = QLabel(self.title.upper())
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color};
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 1px;
            }}
        """)
        title_label.setAlignment(Qt.AlignCenter)

        # Count
        self.count_label = QLabel(str(self.count))
        self.count_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color};
                font-size: 24px;
                font-weight: bold;
            }}
        """)
        self.count_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.count_label)
        self.setLayout(layout)

        # Widget styling
        self.setStyleSheet(f"""
            SeverityCard {{
                background-color: {ProfessionalTheme.PANEL};
                border: 2px solid {self.color};
                border-radius: 8px;
            }}
        """)

        self.setFixedSize(120, 80)

    def update_count(self, count):
        self.count = count
        self.count_label.setText(str(count))

class AnalysisResultsTab(QWidget):
    """Context-aware analysis results tab with detailed anomaly information."""

    def __init__(self, analysis_type, results, data):
        super().__init__()
        self.analysis_type = analysis_type
        self.results = results
        self.data = data
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel(f"{self.analysis_type} Results")
        title_label.setProperty("labelClass", "header")

        timestamp_label = QLabel(f"Generated: {QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')}")
        timestamp_label.setProperty("labelClass", "secondary")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(timestamp_label)

        # Summary Cards
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(16)

        # Count by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for result in self.results:
            severity = result.severity.lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Create severity cards
        severity_colors = {
            'low': ProfessionalTheme.ACCENT_GREEN,
            'medium': ProfessionalTheme.ACCENT_ORANGE,
            'high': ProfessionalTheme.ACCENT_RED,
            'critical': '#8b0000'
        }

        for severity, count in severity_counts.items():
            card = SeverityCard(severity, severity_colors[severity], count)
            summary_layout.addWidget(card)

        summary_layout.addStretch()

        # Total findings
        total_card = SeverityCard("Total", ProfessionalTheme.ACCENT_BLUE, len(self.results))
        summary_layout.addWidget(total_card)

        # Main content area with three-panel splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)

        # Left panel - Results table with detailed anomaly information (40% of space)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        table_label = QLabel("🔍 Detailed Anomaly Analysis")
        table_label.setProperty("labelClass", "subheader")
        left_layout.addWidget(table_label)

        self.results_table = QTableWidget()
        self.setup_detailed_results_table()
        left_layout.addWidget(self.results_table)

        # Middle panel - Context-aware visualizations (35% of space)
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(8, 8, 8, 8)
        middle_layout.setSpacing(8)

        charts_label = QLabel(f"📊 {self.analysis_type} Analysis Overview")
        charts_label.setProperty("labelClass", "subheader")
        middle_layout.addWidget(charts_label)

        # Create context-aware charts
        self.chart_widget = self.create_context_aware_charts()
        self.chart_widget.setMinimumHeight(400)
        middle_layout.addWidget(self.chart_widget)

        # Right panel - Context-specific recommendations (25% of space)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        recommendations_label = QLabel("💡 Context-Aware Recommendations")
        recommendations_label.setProperty("labelClass", "subheader")
        right_layout.addWidget(recommendations_label)

        # Context-specific recommendations
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMinimumHeight(300)
        self.recommendations_text.setPlainText(self.generate_context_recommendations())
        self.recommendations_text.setReadOnly(True)

        # Better styling for recommendations
        self.recommendations_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ProfessionalTheme.SURFACE};
                border: 2px solid {ProfessionalTheme.BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.4;
            }}
        """)

        right_layout.addWidget(self.recommendations_text)

        # Add all panels to splitter with proper proportions
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(middle_widget)
        content_splitter.addWidget(right_widget)

        # Set proportional sizes: 40% table, 35% charts, 25% report
        content_splitter.setSizes([400, 350, 250])
        content_splitter.setStretchFactor(0, 2)  # Table can stretch more
        content_splitter.setStretchFactor(1, 2)  # Charts can stretch more
        content_splitter.setStretchFactor(2, 1)  # Report stretches less

        # Add all to main layout
        layout.addLayout(header_layout)
        layout.addLayout(summary_layout)
        layout.addWidget(content_splitter)

        self.setLayout(layout)

    def setup_detailed_results_table(self):
        """Setup the detailed results table with comprehensive anomaly information and row numbering."""
        if not self.results:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        # Enhanced headers with more detailed information
        headers = ["Anomaly Type", "Description", "Severity", "Confidence", "Risk Score", "Affected Entity", "Timestamp", "Technical Details"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(self.results))
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSortingEnabled(True)

        # Enable and configure row numbers (important: keep visible)
        self.results_table.setVerticalHeaderLabels([str(i + 1) for i in range(len(self.results))])
        self.results_table.verticalHeader().setVisible(True)
        self.results_table.verticalHeader().setDefaultSectionSize(40)
        self.results_table.verticalHeader().setMinimumSectionSize(35)
        self.results_table.verticalHeader().setFixedWidth(60)

        # Set better row height and column width for readability
        self.results_table.horizontalHeader().setDefaultSectionSize(140)
        self.results_table.horizontalHeader().setMinimumSectionSize(100)

        # Populate table with detailed anomaly information
        for row, result in enumerate(self.results):
            # Anomaly Type
            type_item = QTableWidgetItem(result.anomaly_type.replace('_', ' ').title())
            self.results_table.setItem(row, 0, type_item)

            # Description - Enhanced with more details
            desc_item = QTableWidgetItem(result.description)
            desc_item.setToolTip(result.description)  # Full description on hover
            self.results_table.setItem(row, 1, desc_item)

            # Severity - With color coding
            severity_item = QTableWidgetItem(result.severity.upper())
            severity_color = self.get_severity_color(result.severity)
            severity_item.setBackground(QColor(severity_color))
            severity_item.setForeground(QColor("#ffffff"))
            self.results_table.setItem(row, 2, severity_item)

            # Confidence - With percentage
            confidence_item = QTableWidgetItem(f"{result.confidence:.1%}")
            if result.confidence >= 0.8:
                confidence_item.setBackground(QColor(ProfessionalTheme.ACCENT_GREEN))
                confidence_item.setForeground(QColor("#ffffff"))
            elif result.confidence >= 0.6:
                confidence_item.setBackground(QColor(ProfessionalTheme.ACCENT_ORANGE))
                confidence_item.setForeground(QColor("#ffffff"))
            self.results_table.setItem(row, 3, confidence_item)

            # Risk Score - Enhanced calculation
            risk_score = getattr(result, 'risk_score', result.confidence)
            risk_item = QTableWidgetItem(f"{risk_score:.3f}")
            if risk_score >= 0.8:
                risk_item.setBackground(QColor("#8b0000"))
                risk_item.setForeground(QColor("#ffffff"))
            elif risk_score >= 0.6:
                risk_item.setBackground(QColor(ProfessionalTheme.ACCENT_RED))
                risk_item.setForeground(QColor("#ffffff"))
            elif risk_score >= 0.3:
                risk_item.setBackground(QColor(ProfessionalTheme.ACCENT_ORANGE))
                risk_item.setForeground(QColor("#ffffff"))
            self.results_table.setItem(row, 4, risk_item)

            # Affected Entity - Context-aware
            entity = self.get_affected_entity(result)
            entity_item = QTableWidgetItem(entity)
            self.results_table.setItem(row, 5, entity_item)

            # Timestamp - Formatted
            timestamp = getattr(result, 'timestamp', datetime.now())
            timestamp_item = QTableWidgetItem(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            self.results_table.setItem(row, 6, timestamp_item)

            # Technical Details - Context-specific
            technical_details = self.get_technical_details(result)
            tech_item = QTableWidgetItem(technical_details)
            tech_item.setToolTip(technical_details)  # Full details on hover
            self.results_table.setItem(row, 7, tech_item)

        # Resize columns to content but ensure minimum width for readability
        self.results_table.resizeColumnsToContents()

        # Set minimum column widths for better readability
        for col in range(self.results_table.columnCount()):
            current_width = self.results_table.columnWidth(col)
            if current_width < 120:  # Minimum width of 120px
                self.results_table.setColumnWidth(col, 120)

        # Set specific column widths for better display
        self.results_table.setColumnWidth(1, 300)  # Description column wider
        self.results_table.setColumnWidth(7, 280)  # Technical details column wider

    def get_severity_color(self, severity):
        """Get color for severity level."""
        severity_colors = {
            'low': ProfessionalTheme.ACCENT_GREEN,
            'medium': ProfessionalTheme.ACCENT_ORANGE,
            'high': ProfessionalTheme.ACCENT_RED,
            'critical': '#8b0000'
        }
        return severity_colors.get(severity.lower(), ProfessionalTheme.TEXT_MUTED)

    def get_affected_entity(self, result):
        """Get affected entity based on analysis type and result data."""
        # Context-aware entity extraction
        if "Network" in self.analysis_type or "Digital" in self.analysis_type:
            # For network analysis, look for IP addresses
            if hasattr(result, 'source_data') and result.source_data:
                source_ip = result.source_data.get('source_ip')
                dest_ip = result.source_data.get('dest_ip') or result.source_data.get('destination_ip')
                if source_ip:
                    return f"IP: {source_ip}"
                elif dest_ip:
                    return f"Target: {dest_ip}"

        elif "Financial" in self.analysis_type:
            # For financial analysis, look for account information
            if hasattr(result, 'account_id') and result.account_id:
                return f"Account: {result.account_id}"
            elif hasattr(result, 'transaction_id') and result.transaction_id:
                return f"Transaction: {result.transaction_id}"
            elif hasattr(result, 'raw_data') and result.raw_data:
                account = result.raw_data.get('account_id') or result.raw_data.get('user_id')
                if account:
                    return f"Account: {account}"

        # Generic fallback
        if hasattr(result, 'source_data') and result.source_data:
            # Look for any identifier
            for key in ['user_id', 'id', 'entity_id', 'source']:
                if key in result.source_data:
                    return f"{key.replace('_', ' ').title()}: {result.source_data[key]}"

        return "System"

    def get_technical_details(self, result):
        """Get technical details based on analysis type and result data."""
        details = []

        # Context-aware technical details
        if "Network" in self.analysis_type or "Digital" in self.analysis_type:
            if hasattr(result, 'source_data') and result.source_data:
                # Network-specific details
                port = result.source_data.get('port') or result.source_data.get('dest_port')
                if port:
                    details.append(f"Port: {port}")

                protocol = result.source_data.get('protocol')
                if protocol:
                    details.append(f"Protocol: {protocol}")

                bytes_val = result.source_data.get('bytes') or result.source_data.get('bytes_transferred')
                if bytes_val:
                    details.append(f"Data: {bytes_val} bytes")

                threshold = result.source_data.get('threshold')
                if threshold:
                    details.append(f"Threshold: {threshold}")

        elif "Financial" in self.analysis_type:
            # Financial-specific details
            if hasattr(result, 'amount') and result.amount:
                details.append(f"Amount: ${result.amount:,.2f}")

            if hasattr(result, 'raw_data') and result.raw_data:
                # Financial-specific technical data
                transaction_count = result.raw_data.get('transaction_count')
                if transaction_count:
                    details.append(f"Transactions: {transaction_count}")

                velocity_ratio = result.raw_data.get('velocity_ratio')
                if velocity_ratio:
                    details.append(f"Velocity: {velocity_ratio:.2f}x")

                deviation_factor = result.raw_data.get('deviation_factor')
                if deviation_factor:
                    details.append(f"Deviation: {deviation_factor:.2f}x")

        # Add source data information if available
        if hasattr(result, 'raw_data') and result.raw_data:
            for key, value in list(result.raw_data.items())[:2]:  # First 2 items
                if key not in ['amount', 'transaction_count', 'velocity_ratio', 'deviation_factor']:
                    if isinstance(value, (int, float)):
                        details.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        details.append(f"{key.replace('_', ' ').title()}: {value}")

        return " | ".join(details) if details else "No additional technical details"

    def create_context_aware_charts(self):
        """Create charts specific to the analysis type."""
        # Create tabbed widget for different chart types
        chart_tabs = QTabWidget()
        chart_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
        """)

        # Context-specific visualization based on analysis type
        if "Network" in self.analysis_type or "Digital" in self.analysis_type:
            # Network-specific visualizations
            chart_tabs.addTab(self.create_network_analysis_charts(), "Network Analysis")
            chart_tabs.addTab(self.create_network_timeline(), "Network Timeline")
            chart_tabs.addTab(self.create_network_topology(), "Traffic Patterns")

        elif "Financial" in self.analysis_type:
            # Financial-specific visualizations
            chart_tabs.addTab(self.create_financial_analysis_charts(), "Financial Analysis")
            chart_tabs.addTab(self.create_transaction_timeline(), "Transaction Timeline")
            chart_tabs.addTab(self.create_risk_assessment(), "Risk Assessment")

        # Always add general analysis
        chart_tabs.addTab(self.create_general_analysis_charts(), "General Overview")
        chart_tabs.addTab(self.create_anomaly_correlation(), "Anomaly Correlation")

        return chart_tabs

    def create_network_analysis_charts(self):
        """Create network-specific analysis charts."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create matplotlib figure
        fig = Figure(figsize=(10, 8), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        # Set style for dark theme
        plt.style.use('dark_background')

        # Create subplots for network analysis
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # IP Address distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ip_data = []
        for result in self.results:
            if hasattr(result, 'source_data') and result.source_data:
                source_ip = result.source_data.get('source_ip')
                if source_ip:
                    ip_data.append(source_ip)

        if ip_data:
            ip_counts = Counter(ip_data)
            top_ips = ip_counts.most_common(10)
            ips, counts = zip(*top_ips)

            bars = ax1.bar(range(len(counts)), counts, color='#0078d4', alpha=0.7)
            ax1.set_title('Top Suspicious IP Addresses', color='white', fontsize=12, fontweight='bold')
            ax1.set_xlabel('IP Rank', color='white')
            ax1.set_ylabel('Anomaly Count', color='white')
            ax1.tick_params(colors='white')

            # Add IP labels
            for i, ip in enumerate(ips):
                ax1.text(i, counts[i] + max(counts) * 0.01,
                        ip.split('.')[-1], ha='center', va='bottom', color='white', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No IP data available',
                    transform=ax1.transAxes, ha='center', va='center',
                    color='white', fontsize=12)

        # Port distribution
        ax2 = fig.add_subplot(gs[0, 1])
        port_data = []
        for result in self.results:
            if hasattr(result, 'source_data') and result.source_data:
                port = result.source_data.get('port') or result.source_data.get('dest_port')
                if port:
                    port_data.append(port)

        if port_data:
            port_counts = Counter(port_data)
            top_ports = port_counts.most_common(10)
            ports, counts = zip(*top_ports)

            ax2.barh(range(len(counts)), counts, color='#ff8c00', alpha=0.7)
            ax2.set_title('Top Suspicious Ports', color='white', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Anomaly Count', color='white')
            ax2.set_ylabel('Port Rank', color='white')
            ax2.tick_params(colors='white')

            # Add port labels
            for i, port in enumerate(ports):
                ax2.text(counts[i] + max(counts) * 0.01, i,
                        str(port), ha='left', va='center', color='white')
        else:
            ax2.text(0.5, 0.5, 'No port data available',
                    transform=ax2.transAxes, ha='center', va='center',
                    color='white', fontsize=12)

        # Severity vs Confidence scatter
        ax3 = fig.add_subplot(gs[1, :])
        severities = [result.severity for result in self.results]
        confidences = [result.confidence for result in self.results]

        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        severity_numeric = [severity_map.get(s, 0) for s in severities]

        scatter = ax3.scatter(severity_numeric, confidences,
                            c=confidences, cmap='Reds', alpha=0.7, s=100)
        ax3.set_title('Network Anomaly Risk Assessment', color='white', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Severity Level', color='white')
        ax3.set_ylabel('Confidence Score', color='white')
        ax3.set_xticks([1, 2, 3, 4])
        ax3.set_xticklabels(['Low', 'Medium', 'High', 'Critical'])
        ax3.tick_params(colors='white')

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax3)
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Confidence Score', color='white')

        # Create canvas
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #2d2d2d;")
        layout.addWidget(canvas)

        return widget

    def create_financial_analysis_charts(self):
        """Create financial-specific analysis charts."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create matplotlib figure
        fig = Figure(figsize=(10, 8), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        # Set style for dark theme
        plt.style.use('dark_background')

        # Create subplots for financial analysis
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Amount distribution
        ax1 = fig.add_subplot(gs[0, 0])
        amounts = []
        for result in self.results:
            if hasattr(result, 'amount') and result.amount:
                amounts.append(result.amount)

        if amounts:
            ax1.hist(amounts, bins=15, color='#0078d4', alpha=0.7, edgecolor='white')
            ax1.set_title('Suspicious Amount Distribution', color='white', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Transaction Amount ($)', color='white')
            ax1.set_ylabel('Frequency', color='white')
            ax1.tick_params(colors='white')

            # Add mean line
            mean_amount = np.mean(amounts)
            ax1.axvline(mean_amount, color='red', linestyle='--',
                       label=f'Mean: ${mean_amount:,.2f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No amount data available',
                    transform=ax1.transAxes, ha='center', va='center',
                    color='white', fontsize=12)

        # Account distribution
        ax2 = fig.add_subplot(gs[0, 1])
        accounts = []
        for result in self.results:
            if hasattr(result, 'account_id') and result.account_id:
                accounts.append(result.account_id)

        if accounts:
            account_counts = Counter(accounts)
            top_accounts = account_counts.most_common(10)
            account_names, counts = zip(*top_accounts)

            ax2.barh(range(len(counts)), counts, color='#ff8c00', alpha=0.7)
            ax2.set_title('Top Suspicious Accounts', color='white', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Anomaly Count', color='white')
            ax2.set_ylabel('Account Rank', color='white')
            ax2.tick_params(colors='white')

            # Add account labels (truncated)
            for i, account in enumerate(account_names):
                truncated = str(account)[-8:] if len(str(account)) > 8 else str(account)
                ax2.text(counts[i] + max(counts) * 0.01, i,
                        f"...{truncated}", ha='left', va='center', color='white')
        else:
            ax2.text(0.5, 0.5, 'No account data available',
                    transform=ax2.transAxes, ha='center', va='center',
                    color='white', fontsize=12)

        # Risk score vs Amount
        ax3 = fig.add_subplot(gs[1, :])
        risk_scores = []
        plot_amounts = []
        for result in self.results:
            risk_score = getattr(result, 'risk_score', result.confidence)
            amount = getattr(result, 'amount', 0)
            if amount > 0:
                risk_scores.append(risk_score)
                plot_amounts.append(amount)

        if risk_scores and plot_amounts:
            scatter = ax3.scatter(plot_amounts, risk_scores,
                                c=risk_scores, cmap='Reds', alpha=0.7, s=100)
            ax3.set_title('Financial Risk Assessment', color='white', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Transaction Amount ($)', color='white')
            ax3.set_ylabel('Risk Score', color='white')
            ax3.tick_params(colors='white')

            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax3)
            cbar.ax.tick_params(colors='white')
            cbar.set_label('Risk Score', color='white')
        else:
            ax3.text(0.5, 0.5, 'No financial risk data available',
                    transform=ax3.transAxes, ha='center', va='center',
                    color='white', fontsize=12)

        # Create canvas
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #2d2d2d;")
        layout.addWidget(canvas)

        return widget

    def create_general_analysis_charts(self):
        """Create general analysis charts."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create matplotlib figure
        fig = Figure(figsize=(10, 6), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        # Set style for dark theme
        plt.style.use('dark_background')

        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Severity distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        severity_counts = {}
        for result in self.results:
            severity = result.severity.lower()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if severity_counts:
            colors = ['#107c10', '#ff8c00', '#d13438', '#8b0000']
            wedges, texts, autotexts = ax1.pie(severity_counts.values(),
                                              labels=severity_counts.keys(),
                                              autopct='%1.1f%%',
                                              colors=colors[:len(severity_counts)],
                                              textprops={'color': 'white'})
            ax1.set_title('Severity Distribution', color='white', fontsize=12, fontweight='bold')

        # Confidence distribution histogram
        ax2 = fig.add_subplot(gs[0, 1])
        confidences = [result.confidence for result in self.results]
        if confidences:
            ax2.hist(confidences, bins=10, color='#0078d4', alpha=0.7, edgecolor='white')
            ax2.set_title('Confidence Distribution', color='white', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Confidence Score', color='white')
            ax2.set_ylabel('Count', color='white')
            ax2.tick_params(colors='white')

            # Add mean line
            mean_conf = sum(confidences) / len(confidences)
            ax2.axvline(mean_conf, color='red', linestyle='--',
                       label=f'Mean: {mean_conf:.2f}')
            ax2.legend()

        # Anomaly type distribution
        ax3 = fig.add_subplot(gs[1, :])
        anomaly_types = [result.anomaly_type for result in self.results]
        type_counts = Counter(anomaly_types)

        if type_counts:
            top_types = type_counts.most_common(8)  # Top 8 types
            types, counts = zip(*top_types)

            # Clean up type names
            clean_types = [t.replace('_', ' ').title() for t in types]

            bars = ax3.bar(range(len(counts)), counts, color='#0078d4', alpha=0.7)
            ax3.set_title('Top Anomaly Types', color='white', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Anomaly Type', color='white')
            ax3.set_ylabel('Count', color='white')
            ax3.set_xticks(range(len(clean_types)))
            ax3.set_xticklabels(clean_types, rotation=45, ha='right')
            ax3.tick_params(colors='white')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', color='white')

        # Create canvas
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #2d2d2d;")
        layout.addWidget(canvas)

        return widget

    def create_network_timeline(self):
        """Create network-specific timeline visualization."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for network timeline
        label = QLabel("🕒 Network Activity Timeline\n\nNetwork-specific temporal analysis would be displayed here,\nshowing attack patterns, traffic spikes, and incident correlation.")
        label.setStyleSheet("color: white; font-size: 14px; padding: 40px; text-align: center;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return widget

    def create_network_topology(self):
        """Create network topology visualization."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for network topology
        label = QLabel("🌐 Network Traffic Patterns\n\nNetwork topology and traffic flow visualization\nwould be displayed here, showing connection patterns\nand suspicious communication flows.")
        label.setStyleSheet("color: white; font-size: 14px; padding: 40px; text-align: center;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return widget

    def create_transaction_timeline(self):
        """Create financial transaction timeline."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for transaction timeline
        label = QLabel("💰 Transaction Timeline\n\nFinancial transaction temporal analysis would be displayed here,\nshowing transaction patterns, velocity analysis, and fraud correlation.")
        label.setStyleSheet("color: white; font-size: 14px; padding: 40px; text-align: center;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return widget

    def create_risk_assessment(self):
        """Create financial risk assessment visualization."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for risk assessment
        label = QLabel("📊 Financial Risk Assessment\n\nComprehensive financial risk analysis would be displayed here,\nincluding risk scoring, portfolio analysis, and predictive modeling.")
        label.setStyleSheet("color: white; font-size: 14px; padding: 40px; text-align: center;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return widget

    def create_anomaly_correlation(self):
        """Create anomaly correlation analysis."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for correlation analysis
        label = QLabel("🔗 Anomaly Correlation Analysis\n\nCross-anomaly correlation and pattern analysis\nwould be displayed here, showing relationships\nbetween different types of anomalies.")
        label.setStyleSheet("color: white; font-size: 14px; padding: 40px; text-align: center;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return widget

    def generate_context_recommendations(self):
        """Generate context-aware recommendations based on analysis type."""
        recommendations = []

        # Count by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for result in self.results:
            severity = result.severity.lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Context-specific header
        if "Network" in self.analysis_type or "Digital" in self.analysis_type:
            recommendations.append("🛡️ NETWORK SECURITY RECOMMENDATIONS")
        elif "Financial" in self.analysis_type:
            recommendations.append("💼 FINANCIAL SECURITY RECOMMENDATIONS")
        else:
            recommendations.append("🔍 SECURITY ANALYSIS RECOMMENDATIONS")

        recommendations.append("=" * 40)
        recommendations.append("")

        # Analysis summary
        recommendations.append(f"📊 ANALYSIS SUMMARY:")
        recommendations.append(f"Total anomalies detected: {len(self.results)}")
        for severity, count in severity_counts.items():
            if count > 0:
                recommendations.append(f"• {severity.title()}: {count} anomalies")
        recommendations.append("")

        # Priority-based recommendations
        if severity_counts['critical'] > 0:
            recommendations.append("🚨 CRITICAL PRIORITY (Immediate Action Required):")
            recommendations.append(f"• {severity_counts['critical']} critical anomalies require immediate investigation")

            if "Network" in self.analysis_type:
                recommendations.append("• Isolate affected network segments immediately")
                recommendations.append("• Check for lateral movement and privilege escalation")
                recommendations.append("• Review firewall logs and network access controls")
            elif "Financial" in self.analysis_type:
                recommendations.append("• Freeze affected accounts immediately")
                recommendations.append("• Review transaction authorization chains")
                recommendations.append("• Contact financial institutions and regulators")

            recommendations.append("• Initiate incident response procedures")
            recommendations.append("• Document all findings for forensic analysis")
            recommendations.append("")

        if severity_counts['high'] > 0:
            recommendations.append("⚠️ HIGH PRIORITY (24-Hour Response):")
            recommendations.append(f"• {severity_counts['high']} high-severity anomalies need urgent attention")

            if "Network" in self.analysis_type:
                recommendations.append("• Enhance network monitoring and alerting")
                recommendations.append("• Review user access privileges and permissions")
                recommendations.append("• Update security policies and procedures")
            elif "Financial" in self.analysis_type:
                recommendations.append("• Implement additional transaction monitoring")
                recommendations.append("• Review customer due diligence procedures")
                recommendations.append("• Enhance fraud detection algorithms")

            recommendations.append("• Schedule detailed forensic investigation")
            recommendations.append("")

        if severity_counts['medium'] > 0:
            recommendations.append("📋 MEDIUM PRIORITY (Weekly Review):")
            recommendations.append(f"• {severity_counts['medium']} medium-severity anomalies for monitoring")
            recommendations.append("• Implement continuous monitoring for these patterns")
            recommendations.append("• Review and update detection thresholds")
            recommendations.append("• Schedule regular security audits")
            recommendations.append("")

        # Context-specific recommendations
        if "Network" in self.analysis_type:
            recommendations.append("🌐 NETWORK-SPECIFIC ACTIONS:")
            recommendations.append("• Update intrusion detection signatures")
            recommendations.append("• Implement network segmentation where possible")
            recommendations.append("• Review and harden network device configurations")
            recommendations.append("• Enhance endpoint detection and response")

        elif "Financial" in self.analysis_type:
            recommendations.append("💳 FINANCIAL-SPECIFIC ACTIONS:")
            recommendations.append("• Update anti-money laundering controls")
            recommendations.append("• Review transaction velocity limits")
            recommendations.append("• Enhance customer behavior analytics")
            recommendations.append("• Implement real-time fraud scoring")

        recommendations.append("")
        recommendations.append("📈 CONTINUOUS IMPROVEMENT:")
        recommendations.append("• Tune detection algorithms based on findings")
        recommendations.append("• Provide security awareness training")
        recommendations.append("• Update incident response playbooks")
        recommendations.append("• Schedule follow-up analysis in 30 days")

        return "\n".join(recommendations)

class MainWindow(QMainWindow):
    """Inspectra main window with context-aware analysis."""

    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.export_manager = ExportManager()
        self.network_detector = NetworkAnomalyDetector()
        self.fraud_detector = FinancialFraudDetector()

        self.current_data = None
        self.filtered_data = None

        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.apply_theme()

    def setup_ui(self):
        """Setup the main user interface with tabbed layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout - horizontal split
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel - Controls (fixed width)
        left_panel = self.create_left_panel()
        left_panel.setFixedWidth(350)

        # Right panel - Tabbed interface
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        # Create main data tab
        self.main_data_tab = self.create_main_data_tab()
        self.tab_widget.addTab(self.main_data_tab, "Data View")

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.tab_widget)

        # Set stretch factors
        main_layout.setStretchFactor(left_panel, 0)
        main_layout.setStretchFactor(self.tab_widget, 1)

        # Window properties
        self.setWindowTitle("Inspectra - Advanced Forensic Analysis Platform")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Data Loading Section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(8)

        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select dataset file...")
        self.file_path_edit.setReadOnly(True)

        self.browse_button = QPushButton("Browse")
        self.browse_button.setProperty("buttonClass", "primary")
        self.browse_button.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)

        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.setProperty("buttonClass", "success")
        self.load_button.clicked.connect(self.load_data)
        self.load_button.setEnabled(False)

        # Data info
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setProperty("labelClass", "secondary")

        data_layout.addLayout(file_layout)
        data_layout.addWidget(self.load_button)
        data_layout.addWidget(self.data_info_label)

        # Analysis Type Section
        analysis_group = QGroupBox("Forensic Analysis Type")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)

        self.analysis_type_group = QButtonGroup()

        self.no_analysis_radio = QRadioButton("Data View Only")
        self.no_analysis_radio.setChecked(True)
        self.no_analysis_radio.toggled.connect(self.on_analysis_type_changed)

        self.digital_analysis_radio = QRadioButton("Network & Digital Forensics")
        self.digital_analysis_radio.toggled.connect(self.on_analysis_type_changed)

        self.financial_analysis_radio = QRadioButton("Financial Fraud Analysis")
        self.financial_analysis_radio.toggled.connect(self.on_analysis_type_changed)

        self.analysis_type_group.addButton(self.no_analysis_radio)
        self.analysis_type_group.addButton(self.digital_analysis_radio)
        self.analysis_type_group.addButton(self.financial_analysis_radio)

        analysis_layout.addWidget(self.no_analysis_radio)
        analysis_layout.addWidget(self.digital_analysis_radio)
        analysis_layout.addWidget(self.financial_analysis_radio)

        # Analysis Options Section
        self.analysis_options_group = QGroupBox("Analysis Options")
        self.analysis_options_layout = QVBoxLayout(self.analysis_options_group)
        self.analysis_options_group.setVisible(False)

        # Run Analysis Button
        self.run_analysis_button = QPushButton("🔍 Run Forensic Analysis")
        self.run_analysis_button.setProperty("buttonClass", "danger")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.run_analysis_button.setEnabled(False)

        # Data Filtering Section
        filter_group = QGroupBox("Data Filtering")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(8)

        # Global search
        search_layout = QHBoxLayout()
        search_label = QLabel("Global Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search across all columns...")
        self.search_edit.textChanged.connect(self.apply_filters)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)

        # Column filter
        column_layout = QHBoxLayout()
        column_label = QLabel("Column Filter:")
        self.column_combo = QComboBox()
        self.column_combo.currentTextChanged.connect(self.on_column_changed)

        column_layout.addWidget(column_label)
        column_layout.addWidget(self.column_combo)

        # Column value filter
        self.column_filter_edit = QLineEdit()
        self.column_filter_edit.setPlaceholderText("Filter by column value...")
        self.column_filter_edit.textChanged.connect(self.apply_filters)

        # Clear filters button
        self.clear_filters_button = QPushButton("Clear All Filters")
        self.clear_filters_button.setProperty("buttonClass", "warning")
        self.clear_filters_button.clicked.connect(self.clear_filters)

        # Active filters display
        self.active_filters_label = QLabel("No active filters")
        self.active_filters_label.setProperty("labelClass", "muted")
        self.active_filters_label.setWordWrap(True)

        filter_layout.addLayout(search_layout)
        filter_layout.addLayout(column_layout)
        filter_layout.addWidget(self.column_filter_edit)
        filter_layout.addWidget(self.clear_filters_button)
        filter_layout.addWidget(self.active_filters_label)

        # Add all sections to layout
        layout.addWidget(data_group)
        layout.addWidget(analysis_group)
        layout.addWidget(self.analysis_options_group)
        layout.addWidget(self.run_analysis_button)
        layout.addWidget(filter_group)
        layout.addStretch()

        return panel

    def create_main_data_tab(self):
        """Create the main data display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header_label = QLabel("Dataset Overview")
        header_label.setProperty("labelClass", "header")
        layout.addWidget(header_label)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setSortingEnabled(True)

        # Enable and configure row numbers (maintain visibility)
        self.data_table.verticalHeader().setVisible(True)
        self.data_table.verticalHeader().setDefaultSectionSize(40)
        self.data_table.verticalHeader().setMinimumSectionSize(35)
        self.data_table.verticalHeader().setFixedWidth(60)

        # Set better row height for readability
        self.data_table.horizontalHeader().setDefaultSectionSize(120)
        self.data_table.horizontalHeader().setMinimumSectionSize(80)

        layout.addWidget(self.data_table)

        return widget

    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Dataset", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu("Export Data")

        export_csv_action = QAction("Export as CSV", self)
        export_csv_action.triggered.connect(lambda: self.export_data("CSV"))
        export_menu.addAction(export_csv_action)

        export_excel_action = QAction("Export as Excel", self)
        export_excel_action.triggered.connect(lambda: self.export_data("Excel"))
        export_menu.addAction(export_excel_action)

        export_json_action = QAction("Export as JSON", self)
        export_json_action.triggered.connect(lambda: self.export_data("JSON"))
        export_menu.addAction(export_json_action)

        export_pdf_action = QAction("Export PDF Report", self)
        export_pdf_action.triggered.connect(lambda: self.export_data("PDF Report"))
        export_menu.addAction(export_pdf_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        digital_action = QAction("Network & Digital Forensics", self)
        digital_action.triggered.connect(lambda: self.digital_analysis_radio.setChecked(True))
        analysis_menu.addAction(digital_action)

        financial_action = QAction("Financial Fraud Analysis", self)
        financial_action.triggered.connect(lambda: self.financial_analysis_radio.setChecked(True))
        analysis_menu.addAction(financial_action)

        # View menu
        view_menu = menubar.addMenu("View")

        close_all_tabs_action = QAction("Close All Analysis Tabs", self)
        close_all_tabs_action.triggered.connect(self.close_all_analysis_tabs)
        view_menu.addAction(close_all_tabs_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About Inspectra", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = self.statusBar()

        # Status indicators
        self.data_status_label = QLabel("No Data Loaded")
        self.data_status_label.setProperty("labelClass", "secondary")

        self.tab_count_label = QLabel("1 Tab Open")
        self.tab_count_label.setProperty("labelClass", "secondary")

        self.status_bar.addWidget(self.data_status_label)
        self.status_bar.addPermanentWidget(self.tab_count_label)

    def apply_theme(self):
        """Apply the professional theme."""
        self.setStyleSheet(ProfessionalTheme.get_stylesheet())

    def browse_file(self):
        """Browse for a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_button.setEnabled(True)

    def load_data(self):
        """Load the selected data file."""
        file_path = self.file_path_edit.text()
        if not file_path:
            return

        try:
            # Load data
            self.current_data = self.data_loader.load_file(file_path)
            self.filtered_data = self.current_data.copy()

            # Update UI
            self.display_data(self.current_data)
            self.update_column_combo()
            self.update_data_info()
            self.run_analysis_button.setEnabled(True)

            # Update status
            self.data_status_label.setText("Data Loaded Successfully")
            self.data_status_label.setProperty("labelClass", "success")
            self.data_status_label.setStyleSheet(self.data_status_label.styleSheet())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def display_data(self, data):
        """Display data in the main data table with proper row numbering."""
        if data is None or data.empty:
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            return

        # Set table dimensions
        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(len(data.columns))
        self.data_table.setHorizontalHeaderLabels(data.columns.tolist())

        # Enable row numbers in vertical header (IMPORTANT: maintain visibility)
        self.data_table.setVerticalHeaderLabels([str(i + 1) for i in range(len(data))])
        self.data_table.verticalHeader().setVisible(True)
        self.data_table.verticalHeader().setDefaultSectionSize(40)
        self.data_table.verticalHeader().setMinimumSectionSize(35)
        self.data_table.verticalHeader().setFixedWidth(60)

        # Populate table
        for row in range(len(data)):
            for col in range(len(data.columns)):
                value = data.iloc[row, col]
                item = QTableWidgetItem(str(value))

                # Professional color coding for risk/severity columns
                column_name = data.columns[col].lower()
                if 'risk' in column_name or 'severity' in column_name:
                    if isinstance(value, (int, float)):
                        if value >= 0.8 or (isinstance(value, str) and 'critical' in str(value).lower()):
                            item.setBackground(QColor("#8b0000"))
                            item.setForeground(QColor("#ffffff"))
                        elif value >= 0.6 or (isinstance(value, str) and 'high' in str(value).lower()):
                            item.setBackground(QColor(ProfessionalTheme.ACCENT_RED))
                            item.setForeground(QColor("#ffffff"))
                        elif value >= 0.3 or (isinstance(value, str) and 'medium' in str(value).lower()):
                            item.setBackground(QColor(ProfessionalTheme.ACCENT_ORANGE))
                            item.setForeground(QColor("#ffffff"))
                        else:
                            item.setBackground(QColor(ProfessionalTheme.ACCENT_GREEN))
                            item.setForeground(QColor("#ffffff"))

                self.data_table.setItem(row, col, item)

        # Resize columns to content but ensure minimum width for readability
        self.data_table.resizeColumnsToContents()

        # Set minimum column widths for better readability
        for col in range(self.data_table.columnCount()):
            current_width = self.data_table.columnWidth(col)
            if current_width < 100:  # Minimum width of 100px
                self.data_table.setColumnWidth(col, 100)

        # Ensure row numbers column has proper width
        self.data_table.verticalHeader().setFixedWidth(60)

    def update_column_combo(self):
        """Update the column filter combo box."""
        if self.current_data is not None:
            self.column_combo.clear()
            self.column_combo.addItem("Select column...")
            self.column_combo.addItems(self.current_data.columns.tolist())

    def update_data_info(self):
        """Update the data information display."""
        if self.current_data is not None:
            rows = len(self.current_data)
            cols = len(self.current_data.columns)
            self.data_info_label.setText(f"Dataset: {rows:,} rows, {cols} columns")
            self.data_info_label.setProperty("labelClass", "success")
        else:
            self.data_info_label.setText("No data loaded")
            self.data_info_label.setProperty("labelClass", "secondary")

        self.data_info_label.setStyleSheet(self.data_info_label.styleSheet())

    def on_analysis_type_changed(self):
        """Handle analysis type change."""
        if self.no_analysis_radio.isChecked():
            self.analysis_options_group.setVisible(False)
            self.run_analysis_button.setText("🔍 Run Analysis")
        else:
            self.analysis_options_group.setVisible(True)
            self.setup_analysis_options()

    def setup_analysis_options(self):
        """Setup analysis options based on selected type."""
        # Clear existing options
        for i in reversed(range(self.analysis_options_layout.count())):
            self.analysis_options_layout.itemAt(i).widget().setParent(None)

        if self.digital_analysis_radio.isChecked():
            self.setup_digital_analysis_options()
            self.run_analysis_button.setText("🛡️ Run Network Forensics")
        elif self.financial_analysis_radio.isChecked():
            self.setup_financial_analysis_options()
            self.run_analysis_button.setText("💼 Run Financial Analysis")

    def setup_digital_analysis_options(self):
        """Setup digital forensic analysis options."""
        label = QLabel("Network Anomaly Detection Methods:")
        label.setProperty("labelClass", "subheader")
        self.analysis_options_layout.addWidget(label)

        self.digital_options = {}
        options = [
            ("port_scanning", "Port Scanning Detection"),
            ("ddos_attacks", "DDoS Attack Detection"),
            ("unusual_traffic", "Unusual Traffic Patterns"),
            ("suspicious_connections", "Suspicious Connections"),
            ("data_exfiltration", "Data Exfiltration Detection"),
            ("lateral_movement", "Lateral Movement Detection"),
            ("dns_tunneling", "DNS Tunneling Detection"),
            ("malware_beaconing", "Malware Beaconing Detection")
        ]

        for key, text in options:
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            self.digital_options[key] = checkbox
            self.analysis_options_layout.addWidget(checkbox)

    def setup_financial_analysis_options(self):
        """Setup financial forensic analysis options."""
        label = QLabel("Financial Fraud Detection Methods:")
        label.setProperty("labelClass", "subheader")
        self.analysis_options_layout.addWidget(label)

        self.financial_options = {}
        options = [
            ("unusual_amounts", "Unusual Amount Detection"),
            ("velocity_fraud", "Velocity Fraud Detection"),
            ("account_takeover", "Account Takeover Detection"),
            ("money_laundering", "Money Laundering Detection"),
            ("card_fraud", "Card Fraud Detection"),
            ("geographic_anomalies", "Geographic Anomalies"),
            ("merchant_fraud", "Merchant Fraud Detection"),
            ("structuring", "Transaction Structuring Detection")
        ]

        for key, text in options:
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            self.financial_options[key] = checkbox
            self.analysis_options_layout.addWidget(checkbox)

    def run_analysis(self):
        """Run the selected analysis and open results in new tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        try:
            if self.digital_analysis_radio.isChecked():
                results = self.run_digital_analysis()
                analysis_type = "Network & Digital Forensic Analysis"
            elif self.financial_analysis_radio.isChecked():
                results = self.run_financial_analysis()
                analysis_type = "Financial Fraud Analysis"
            else:
                return

            # Create new tab with context-aware results
            self.create_analysis_tab(analysis_type, results)

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {str(e)}")

    def run_digital_analysis(self):
        """Run digital forensic analysis."""
        # Get selected methods
        selected_methods = []
        for key, checkbox in self.digital_options.items():
            if checkbox.isChecked():
                selected_methods.append(key)

        # Run analysis
        return self.network_detector.analyze_network_data(
            self.current_data, selected_methods
        )

    def run_financial_analysis(self):
        """Run financial forensic analysis."""
        # Get selected methods
        selected_methods = []
        for key, checkbox in self.financial_options.items():
            if checkbox.isChecked():
                selected_methods.append(key)

        # Run analysis
        return self.fraud_detector.analyze_financial_data(
            self.current_data, selected_methods
        )

    def create_analysis_tab(self, analysis_type, results):
        """Create a new tab with context-aware analysis results."""
        # Create results tab with context awareness
        results_tab = AnalysisResultsTab(analysis_type, results, self.current_data)

        # Add tab with close button
        tab_index = self.tab_widget.addTab(results_tab, f"{analysis_type}")

        # Switch to the new tab
        self.tab_widget.setCurrentIndex(tab_index)

        # Update tab count
        self.update_tab_count()

    def close_tab(self, index):
        """Close a tab."""
        if index > 0:  # Don't close the main data tab
            self.tab_widget.removeTab(index)
            self.update_tab_count()

    def close_all_analysis_tabs(self):
        """Close all analysis tabs except the main data tab."""
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        self.update_tab_count()

    def update_tab_count(self):
        """Update the tab count in status bar."""
        count = self.tab_widget.count()
        self.tab_count_label.setText(f"{count} Tab{'s' if count != 1 else ''} Open")

    def on_column_changed(self):
        """Handle column selection change."""
        column = self.column_combo.currentText()
        if column and column != "Select column...":
            self.column_filter_edit.setPlaceholderText(f"Filter {column}...")
        else:
            self.column_filter_edit.setPlaceholderText("Filter by column value...")

    def apply_filters(self):
        """Apply current filters to the data."""
        if self.current_data is None:
            return

        filtered_data = self.current_data.copy()
        active_filters = []

        # Global search filter
        search_text = self.search_edit.text().strip()
        if search_text:
            mask = filtered_data.astype(str).apply(
                lambda x: x.str.contains(search_text, case=False, na=False)
            ).any(axis=1)
            filtered_data = filtered_data[mask]
            active_filters.append(f"Global: '{search_text}'")

        # Column-specific filter
        column = self.column_combo.currentText()
        column_filter = self.column_filter_edit.text().strip()
        if column and column != "Select column..." and column_filter:
            if column in filtered_data.columns:
                mask = filtered_data[column].astype(str).str.contains(
                    column_filter, case=False, na=False
                )
                filtered_data = filtered_data[mask]
                active_filters.append(f"{column}: '{column_filter}'")

        # Update display
        self.filtered_data = filtered_data
        self.display_data(filtered_data)

        # Update active filters display
        if active_filters:
            self.active_filters_label.setText("Active filters: " + ", ".join(active_filters))
            self.active_filters_label.setProperty("labelClass", "warning")
        else:
            self.active_filters_label.setText("No active filters")
            self.active_filters_label.setProperty("labelClass", "muted")

        self.active_filters_label.setStyleSheet(self.active_filters_label.styleSheet())

    def clear_filters(self):
        """Clear all filters."""
        self.search_edit.clear()
        self.column_combo.setCurrentIndex(0)
        self.column_filter_edit.clear()

        if self.current_data is not None:
            self.filtered_data = self.current_data.copy()
            self.display_data(self.filtered_data)

    def export_data(self, format_type):
        """Export the current data."""
        if self.filtered_data is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        try:
            if format_type == "CSV":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export CSV", "", "CSV Files (*.csv)"
                )
                if file_path:
                    self.export_manager.export_to_csv(self.filtered_data, file_path)

            elif format_type == "Excel":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export Excel", "", "Excel Files (*.xlsx)"
                )
                if file_path:
                    self.export_manager.export_to_excel(self.filtered_data, file_path)

            elif format_type == "JSON":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export JSON", "", "JSON Files (*.json)"
                )
                if file_path:
                    self.export_manager.export_to_json(self.filtered_data, file_path)

            elif format_type == "PDF Report":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export PDF Report", "", "PDF Files (*.pdf)"
                )
                if file_path:
                    # Get current analysis results if any
                    current_tab = self.tab_widget.currentWidget()
                    analysis_results = None
                    if isinstance(current_tab, AnalysisResultsTab):
                        analysis_results = current_tab.results

                    self.export_manager.export_to_pdf(
                        self.filtered_data, file_path, analysis_results
                    )

            if 'file_path' in locals() and file_path:
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Inspectra",
            """
            <h3>Inspectra - Advanced Forensic Analysis Platform</h3>
            <p><b>Context-Aware Analysis Edition</b></p>
            <p>Professional forensic analysis tool with intelligent, context-aware visualizations and detailed anomaly reporting.</p>
            <p>Features:</p>
            <ul>
                <li>Context-aware forensic analysis (Network & Financial)</li>
                <li>Intelligent visualization based on analysis type</li>
                <li>Detailed anomaly information with row-level tracking</li>
                <li>Advanced threat detection algorithms</li>
                <li>Interactive data visualization and analysis</li>
                <li>Comprehensive filtering and search capabilities</li>
                <li>Multiple export formats with detailed reporting</li>
                <li>Professional interface optimized for forensic investigators</li>
            </ul>
            <p>Version 3.0 - Inspectra Context-Aware Edition</p>
            """
        )

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Inspectra")
    app.setApplicationVersion("3.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()