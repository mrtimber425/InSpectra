# main_gui.py
# Main InSpectra Analytics GUI Application

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import warnings
import webbrowser
import tempfile

# Import our custom modules
from data_loader import DataLoader
from data_processor import DataProcessor
from analytics_engine import AnalyticsEngine
from chart_generator import ChartGenerator
from ml_engine import MLEngine
from export_manager import ExportManager
from database_manager import DatabaseManager

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')


class InSpectraGUI:
    """Main InSpectra Analytics GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("InSpectra Analytics")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)

        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.analytics_engine = AnalyticsEngine()
        self.chart_generator = ChartGenerator()
        self.ml_engine = MLEngine()
        self.export_manager = ExportManager()
        self.database_manager = DatabaseManager()

        # Initialize data storage
        self.current_data = None
        self.original_data = None
        self.data_info = {}
        self.result_queue = queue.Queue()
        self.analysis_history = []
        self.export_history = []

        # Chart storage
        self.charts = {}
        self.current_chart_id = 0

        # Setup styles and create GUI
        self.setup_styles()
        self.create_widgets()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Bind cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Configure GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Success.TLabel', foreground='#27ae60', font=('Arial', 10, 'bold'))
        style.configure('Error.TLabel', foreground='#e74c3c', font=('Arial', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='#f39c12', font=('Arial', 10, 'bold'))
        style.configure('Info.TLabel', foreground='#3498db', font=('Arial', 10))

    def create_widgets(self):
        """Create all GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title section
        self.create_title_section(main_frame)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create all tabs
        self.create_data_tab()
        self.create_preprocessing_tab()
        self.create_analytics_tab()
        self.create_charts_tab()
        self.create_ml_tab()
        self.create_export_tab()
        self.create_database_tab()
        self.create_settings_tab()

        # Status bar
        self.create_status_bar(main_frame)

    def create_title_section(self, parent):
        """Create title section"""
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        title_frame.grid_columnconfigure(0, weight=1)

        title_label = ttk.Label(title_frame, text="🚀 InSpectra Analytics Platform v2.0",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0)

        subtitle_label = ttk.Label(title_frame, text="Advanced Data Processing & Analytics Suite",
                                   style='Info.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(2, 0))

    def create_data_tab(self):
        """Create data management tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📁 Data Manager")

        data_frame.grid_rowconfigure(3, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)

        # Data source section
        source_frame = ttk.LabelFrame(data_frame, text="Data Sources", padding="15")
        source_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        source_frame.grid_columnconfigure(1, weight=1)

        # File loading
        ttk.Label(source_frame, text="📂 Load File:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))

        file_frame = ttk.Frame(source_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Path:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(file_frame, text="Load", command=self.load_file).grid(row=0, column=3, padx=(5, 0))

        # Sample data section
        ttk.Label(source_frame, text="📊 Sample Data:", style='Header.TLabel').grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))

        sample_frame = ttk.Frame(source_frame)
        sample_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        sample_datasets = ["Iris", "Titanic", "Boston Housing", "Wine Quality", "Stock Market"]
        for i, dataset in enumerate(sample_datasets):
            ttk.Button(sample_frame, text=dataset,
                       command=lambda d=dataset: self.load_sample_data(d)).grid(row=0, column=i, padx=(0, 5))

        # URL loading
        ttk.Label(source_frame, text="🌐 Load from URL:", style='Header.TLabel').grid(
            row=4, column=0, sticky=tk.W, pady=(0, 5))

        url_frame = ttk.Frame(source_frame)
        url_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))

        ttk.Button(url_frame, text="Load CSV from URL",
                   command=self.load_url_dialog).grid(row=0, column=0)

        # Data info section
        self.create_data_info_section(data_frame)

        # Data operations
        self.create_data_operations_section(data_frame)

        # Data preview
        self.create_data_preview_section(data_frame)

    def create_data_info_section(self, parent):
        """Create data information section"""
        info_frame = ttk.LabelFrame(parent, text="Dataset Information", padding="15")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.grid_columnconfigure(1, weight=1)
        info_frame.grid_columnconfigure(3, weight=1)

        self.info_labels = {}
        info_items = [
            ("📊 Rows:", "rows", 0, 0),
            ("📈 Columns:", "columns", 0, 2),
            ("💾 Memory:", "memory", 1, 0),
            ("📁 Source:", "filename", 1, 2),
            ("🕒 Loaded:", "load_time", 2, 0),
            ("📏 Size:", "file_size", 2, 2)
        ]

        for label, key, row, col in info_items:
            ttk.Label(info_frame, text=label, style='Header.TLabel').grid(
                row=row, column=col, sticky=tk.W, padx=(0, 10), pady=2)
            self.info_labels[key] = ttk.Label(info_frame, text="N/A", style='Info.TLabel')
            self.info_labels[key].grid(row=row, column=col + 1, sticky=tk.W, padx=(0, 30), pady=2)

    def create_data_operations_section(self, parent):
        """Create data operations section"""
        ops_frame = ttk.LabelFrame(parent, text="Data Operations", padding="15")
        ops_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(ops_frame, text="🔄 Refresh View",
                   command=self.refresh_preview).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(ops_frame, text="📋 Copy to Clipboard",
                   command=self.copy_to_clipboard).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(ops_frame, text="🧹 Reset Data",
                   command=self.reset_data).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(ops_frame, text="ℹ️ Data Info",
                   command=self.show_data_info).grid(row=0, column=3, padx=(0, 5))

    def create_data_preview_section(self, parent):
        """Create data preview section"""
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding="15")
        preview_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Preview controls
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(preview_controls, text="Rows:").grid(row=0, column=0, sticky=tk.W)
        self.preview_rows_var = tk.StringVar(value="1000")
        ttk.Entry(preview_controls, textvariable=self.preview_rows_var, width=10).grid(
            row=0, column=1, padx=(5, 0))

        ttk.Button(preview_controls, text="Refresh",
                   command=self.refresh_preview).grid(row=0, column=2, padx=(10, 0))

        # Data table
        self.create_data_table(preview_frame)

    def create_data_table(self, parent):
        """Create data table with scrollbars"""
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Create Treeview
        self.data_tree = ttk.Treeview(table_frame, show='headings')
        self.data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)

    def create_preprocessing_tab(self):
        """Create data preprocessing tab"""
        preprocess_frame = ttk.Frame(self.notebook)
        self.notebook.add(preprocess_frame, text="🔧 Preprocessing")

        preprocess_frame.grid_rowconfigure(2, weight=1)
        preprocess_frame.grid_columnconfigure(0, weight=1)

        # Quality assessment
        quality_frame = ttk.LabelFrame(preprocess_frame, text="Data Quality Assessment", padding="15")
        quality_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(quality_frame, text="📊 Generate Quality Report",
                   command=self.generate_quality_report).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(quality_frame, text="🧹 Auto Clean Data",
                   command=self.auto_clean_data).grid(row=0, column=1, padx=(0, 10))

        # Preprocessing operations
        operations_frame = ttk.LabelFrame(preprocess_frame, text="Data Operations", padding="15")
        operations_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Missing values
        missing_frame = ttk.Frame(operations_frame)
        missing_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(missing_frame, text="Missing Values:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.missing_strategy_var = tk.StringVar(value="auto")
        missing_combo = ttk.Combobox(missing_frame, textvariable=self.missing_strategy_var,
                                     values=["auto", "drop_rows", "drop_columns", "mean", "median", "mode"])
        missing_combo.grid(row=0, column=1, padx=(10, 0))
        ttk.Button(missing_frame, text="Apply",
                   command=self.handle_missing_values).grid(row=0, column=2, padx=(10, 0))

        # Duplicates
        duplicate_frame = ttk.Frame(operations_frame)
        duplicate_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(duplicate_frame, text="Duplicates:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Button(duplicate_frame, text="Remove Duplicates",
                   command=self.remove_duplicates).grid(row=0, column=1, padx=(10, 0))

        # Outliers
        outlier_frame = ttk.Frame(operations_frame)
        outlier_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(outlier_frame, text="Outliers:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.outlier_method_var = tk.StringVar(value="iqr")
        outlier_combo = ttk.Combobox(outlier_frame, textvariable=self.outlier_method_var,
                                     values=["iqr", "zscore"])
        outlier_combo.grid(row=0, column=1, padx=(10, 0))
        ttk.Button(outlier_frame, text="Handle Outliers",
                   command=self.handle_outliers).grid(row=0, column=2, padx=(10, 0))

        # Results display
        results_frame = ttk.LabelFrame(preprocess_frame, text="Processing Results", padding="15")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        self.preprocessing_results = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15)
        self.preprocessing_results.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_analytics_tab(self):
        """Create analytics tab"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="📊 Analytics")

        analytics_frame.grid_rowconfigure(1, weight=1)
        analytics_frame.grid_columnconfigure(0, weight=1)

        # Analytics controls
        controls_frame = ttk.LabelFrame(analytics_frame, text="Analytics Controls", padding="15")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Analysis type selection
        ttk.Label(controls_frame, text="Analysis Type:").grid(row=0, column=0, sticky=tk.W)
        self.analysis_type_var = tk.StringVar(value="Summary Statistics")
        analysis_combo = ttk.Combobox(controls_frame, textvariable=self.analysis_type_var,
                                      values=["Summary Statistics", "Correlation Analysis",
                                              "Group Analysis", "Time Series Analysis",
                                              "Distribution Analysis", "Association Analysis"])
        analysis_combo.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))
        analysis_combo.bind('<<ComboboxSelected>>', self.on_analysis_type_change)

        ttk.Button(controls_frame, text="Run Analysis",
                   command=self.run_analysis).grid(row=0, column=2, padx=(10, 0))

        # Column selection frame (initially hidden)
        self.column_frame = ttk.LabelFrame(analytics_frame, text="Column Selection", padding="15")

        # Results area
        results_frame = ttk.LabelFrame(analytics_frame, text="Analysis Results", padding="15")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_charts_tab(self):
        """Create charts tab"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="📈 Charts")

        charts_frame.grid_rowconfigure(1, weight=1)
        charts_frame.grid_columnconfigure(0, weight=1)

        # Chart controls
        self.create_chart_controls(charts_frame)

        # Chart display area
        self.create_chart_display(charts_frame)

    def create_chart_controls(self, parent):
        """Create chart controls"""
        chart_controls = ttk.LabelFrame(parent, text="Chart Controls", padding="15")
        chart_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Chart type selection
        ttk.Label(chart_controls, text="Chart Type:").grid(row=0, column=0, sticky=tk.W)
        self.chart_type_var = tk.StringVar(value="histogram")
        chart_combo = ttk.Combobox(chart_controls, textvariable=self.chart_type_var,
                                   values=["histogram", "bar", "line", "scatter", "box",
                                           "violin", "correlation_heatmap", "distribution"])
        chart_combo.grid(row=0, column=1, padx=(5, 0))
        chart_combo.bind('<<ComboboxSelected>>', self.on_chart_type_change)

        # Column selections
        ttk.Label(chart_controls, text="X Column:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.x_column_var = tk.StringVar()
        self.x_column_combo = ttk.Combobox(chart_controls, textvariable=self.x_column_var)
        self.x_column_combo.grid(row=0, column=3, padx=(5, 0))

        ttk.Label(chart_controls, text="Y Column:").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.y_column_var = tk.StringVar()
        self.y_column_combo = ttk.Combobox(chart_controls, textvariable=self.y_column_var)
        self.y_column_combo.grid(row=0, column=5, padx=(5, 0))

        ttk.Button(chart_controls, text="Generate Chart",
                   command=self.generate_chart).grid(row=0, column=6, padx=(20, 0))

    def create_chart_display(self, parent):
        """Create chart display area"""
        chart_display = ttk.LabelFrame(parent, text="Chart Display", padding="15")
        chart_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_display.grid_rowconfigure(0, weight=1)
        chart_display.grid_columnconfigure(0, weight=1)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, chart_display)
        self.chart_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Navigation toolbar
        toolbar_frame = ttk.Frame(chart_display)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.toolbar = NavigationToolbar2Tk(self.chart_canvas, toolbar_frame)
        self.toolbar.update()

        # Chart control buttons
        button_frame = ttk.Frame(chart_display)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="Save Chart",
                   command=self.save_chart).grid(row=0, column=0)
        ttk.Button(button_frame, text="Clear Chart",
                   command=self.clear_chart).grid(row=0, column=1, padx=(5, 0))

    def create_ml_tab(self):
        """Create machine learning tab"""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 Machine Learning")

        ml_frame.grid_rowconfigure(1, weight=1)
        ml_frame.grid_columnconfigure(0, weight=1)

        # ML controls
        ml_controls = ttk.LabelFrame(ml_frame, text="Machine Learning Controls", padding="15")
        ml_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Task type
        ttk.Label(ml_controls, text="Task Type:").grid(row=0, column=0, sticky=tk.W)
        self.ml_task_var = tk.StringVar(value="Classification")
        task_combo = ttk.Combobox(ml_controls, textvariable=self.ml_task_var,
                                  values=["Classification", "Regression", "Clustering", "PCA", "Auto ML"])
        task_combo.grid(row=0, column=1, padx=(5, 0))

        # Target column
        ttk.Label(ml_controls, text="Target Column:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.target_column_var = tk.StringVar()
        self.target_column_combo = ttk.Combobox(ml_controls, textvariable=self.target_column_var)
        self.target_column_combo.grid(row=0, column=3, padx=(5, 0))

        ttk.Button(ml_controls, text="Run ML Task",
                   command=self.run_ml_task).grid(row=0, column=4, padx=(20, 0))

        # ML results
        ml_results_frame = ttk.LabelFrame(ml_frame, text="ML Results", padding="15")
        ml_results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        ml_results_frame.grid_rowconfigure(0, weight=1)
        ml_results_frame.grid_columnconfigure(0, weight=1)

        self.ml_results_text = scrolledtext.ScrolledText(ml_results_frame, wrap=tk.WORD, height=20)
        self.ml_results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_export_tab(self):
        """Create export tab"""
        export_frame = ttk.Frame(self.notebook)
        self.notebook.add(export_frame, text="📤 Export")

        export_frame.grid_rowconfigure(1, weight=1)
        export_frame.grid_columnconfigure(0, weight=1)

        # Export controls
        export_controls = ttk.LabelFrame(export_frame, text="Export Options", padding="15")
        export_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Data export
        ttk.Label(export_controls, text="Export Data:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)

        data_export_frame = ttk.Frame(export_controls)
        data_export_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        self.export_format_var = tk.StringVar(value="csv")
        formats = ["csv", "excel", "json", "parquet", "html"]
        for i, fmt in enumerate(formats):
            ttk.Radiobutton(data_export_frame, text=fmt.upper(), variable=self.export_format_var,
                            value=fmt).grid(row=0, column=i, sticky=tk.W, padx=(0, 10))

        ttk.Button(data_export_frame, text="Export Data",
                   command=self.export_data).grid(row=0, column=len(formats), padx=(20, 0))

        # Report generation
        ttk.Label(export_controls, text="Generate Report:", style='Header.TLabel').grid(
            row=2, column=0, sticky=tk.W, pady=(20, 0))

        report_frame = ttk.Frame(export_controls)
        report_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(report_frame, text="Summary Report",
                   command=self.generate_summary_report).grid(row=0, column=0)
        ttk.Button(report_frame, text="Analysis Report",
                   command=self.generate_analysis_report).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(report_frame, text="Dashboard",
                   command=self.generate_dashboard).grid(row=0, column=2, padx=(10, 0))

        # Report display
        report_display = ttk.LabelFrame(export_frame, text="Report Preview", padding="15")
        report_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        report_display.grid_rowconfigure(0, weight=1)
        report_display.grid_columnconfigure(0, weight=1)

        self.report_text = scrolledtext.ScrolledText(report_display, wrap=tk.WORD, height=20)
        self.report_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_database_tab(self):
        """Create database tab"""
        db_frame = ttk.Frame(self.notebook)
        self.notebook.add(db_frame, text="🗄️ Database")

        db_frame.grid_rowconfigure(1, weight=1)
        db_frame.grid_columnconfigure(0, weight=1)

        # Database controls
        db_controls = ttk.LabelFrame(db_frame, text="Database Connections", padding="15")
        db_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(db_controls, text="Connect to Database",
                   command=self.show_database_dialog).grid(row=0, column=0)
        ttk.Button(db_controls, text="Execute Query",
                   command=self.show_query_dialog).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(db_controls, text="List Tables",
                   command=self.list_database_tables).grid(row=0, column=2, padx=(10, 0))

        # Database info
        db_info_frame = ttk.LabelFrame(db_frame, text="Database Information", padding="15")
        db_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        db_info_frame.grid_rowconfigure(0, weight=1)
        db_info_frame.grid_columnconfigure(0, weight=1)

        self.db_info_text = scrolledtext.ScrolledText(db_info_frame, wrap=tk.WORD, height=20)
        self.db_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")

        # Application settings
        app_settings = ttk.LabelFrame(settings_frame, text="Application Settings", padding="15")
        app_settings.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(app_settings, text="Theme:").grid(row=0, column=0, sticky=tk.W)
        self.theme_var = tk.StringVar(value="clam")
        theme_combo = ttk.Combobox(app_settings, textvariable=self.theme_var,
                                   values=["clam", "alt", "default", "classic"])
        theme_combo.grid(row=0, column=1, padx=(5, 0))
        theme_combo.bind('<<ComboboxSelected>>', self.change_theme)

        # About section
        about_frame = ttk.LabelFrame(settings_frame, text="About", padding="15")
        about_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        about_text = """
InSpectra Analytics 

A comprehensive data analytics and visualization tool built with Python.

Features:
• Advanced data loading and preprocessing
• Statistical analysis and machine learning
• Interactive data visualization
• Database connectivity
• Report generation and export

Built with: Tkinter, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
P.S This is a personal project and not an enterprise application!!!
        """

        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).grid(row=0, column=0, sticky=(tk.W, tk.N))

    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                            mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=1, sticky=tk.E)

    # =============================================================================
    # Data Loading Methods
    # =============================================================================

    def browse_file(self):
        """Browse for data file"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Parquet files", "*.parquet"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def load_file(self):
        """Load data from file"""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file first")
            return

        self.status_var.set("Loading file...")
        self.progress_var.set(25)
        self.root.update()

        threading.Thread(target=self._load_file_thread, args=(file_path,), daemon=True).start()

    def _load_file_thread(self, file_path):
        """Load file in background thread"""
        try:
            self.progress_var.set(50)
            df, info = self.data_loader.load_file(file_path)

            self.current_data = df
            self.original_data = df.copy()
            self.data_info = info

            # Register with analytics engine
            self.analytics_engine.register_dataframe(df)

            self.progress_var.set(100)
            self.result_queue.put(("success", "File loaded successfully"))

        except Exception as e:
            self.result_queue.put(("error", str(e)))

        self.root.after(100, self.check_results)

    def load_sample_data(self, dataset_name):
        """Load sample dataset"""
        try:
            self.status_var.set(f"Loading {dataset_name} dataset...")
            df, info = self.data_loader.get_sample_data(dataset_name)

            self.current_data = df
            self.original_data = df.copy()
            self.data_info = info

            # Register with analytics engine
            self.analytics_engine.register_dataframe(df)

            self.update_data_display()
            self.status_var.set(f"{dataset_name} dataset loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")

    def load_url_dialog(self):
        """Show URL loading dialog"""
        url = simpledialog.askstring("Load from URL", "Enter CSV URL:")
        if url:
            try:
                self.status_var.set("Loading from URL...")
                df, info = self.data_loader.load_from_url(url), {"filename": "URL Data", "rows": 0, "columns": 0}

                # Update info
                info['rows'] = len(df)
                info['columns'] = len(df.columns)
                info['memory'] = f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB"
                info['load_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                self.current_data = df
                self.original_data = df.copy()
                self.data_info = info

                # Register with analytics engine
                self.analytics_engine.register_dataframe(df)

                self.update_data_display()
                self.status_var.set("URL data loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load from URL: {str(e)}")

    def check_results(self):
        """Check for results from background threads"""
        try:
            while True:
                result_type, message = self.result_queue.get_nowait()

                if result_type == "success":
                    self.status_var.set(message)
                    self.update_data_display()
                elif result_type == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Error", message)

                self.progress_var.set(0)

        except queue.Empty:
            pass

    def update_data_display(self):
        """Update data display after loading"""
        if self.current_data is None:
            return

        # Update info labels
        for key, label in self.info_labels.items():
            if key in self.data_info:
                label.config(text=str(self.data_info[key]))

        # Update data preview
        self.refresh_preview()

        # Update column selections
        self.update_column_selections()

    def refresh_preview(self):
        """Refresh data preview table"""
        if self.current_data is None:
            return

        try:
            max_rows = int(self.preview_rows_var.get())
        except ValueError:
            max_rows = 1000

        # Clear existing data
        self.data_tree.delete(*self.data_tree.get_children())

        # Set up columns
        columns = list(self.current_data.columns)
        self.data_tree["columns"] = columns

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)

        # Add data rows
        display_data = self.current_data.head(max_rows)
        for index, row in display_data.iterrows():
            values = [str(val) if pd.notna(val) else "" for val in row]
            self.data_tree.insert("", "end", values=values)

    def update_column_selections(self):
        """Update column combo boxes"""
        if self.current_data is None:
            return

        columns = list(self.current_data.columns)

        # Update chart column selections
        self.x_column_combo['values'] = columns
        self.y_column_combo['values'] = columns

        # Update ML target column selection
        self.target_column_combo['values'] = columns

        if columns:
            self.x_column_var.set(columns[0])
            if len(columns) > 1:
                self.y_column_var.set(columns[1])
                self.target_column_var.set(columns[-1])  # Last column as default target

    def copy_to_clipboard(self):
        """Copy data to clipboard"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            self.current_data.to_clipboard(index=False)
            messagebox.showinfo("Success", "Data copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy data: {str(e)}")

    def reset_data(self):
        """Reset data to original state"""
        if self.original_data is None:
            messagebox.showerror("Error", "No original data to reset to")
            return

        self.current_data = self.original_data.copy()
        self.analytics_engine.register_dataframe(self.current_data)
        self.update_data_display()
        self.status_var.set("Data reset to original state")

    def show_data_info(self):
        """Show detailed data information"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        info_window = tk.Toplevel(self.root)
        info_window.title("Data Information")
        info_window.geometry("600x400")

        info_text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Generate detailed info
        info_str = f"Dataset Information\n{'=' * 50}\n\n"
        info_str += f"Shape: {self.current_data.shape}\n"
        info_str += f"Memory Usage: {self.current_data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB\n\n"

        info_str += "Column Information:\n"
        for col in self.current_data.columns:
            info_str += f"\n{col}:\n"
            info_str += f"  Data Type: {self.current_data[col].dtype}\n"
            info_str += f"  Non-Null Count: {self.current_data[col].count()}\n"
            info_str += f"  Null Count: {self.current_data[col].isnull().sum()}\n"
            info_str += f"  Unique Values: {self.current_data[col].nunique()}\n"

        info_text.insert(tk.END, info_str)

    # =============================================================================
    # Preprocessing Methods
    # =============================================================================

    def generate_quality_report(self):
        """Generate data quality report"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            report = self.data_processor.get_data_quality_report(self.current_data)

            # Format report for display
            report_text = "DATA QUALITY REPORT\n" + "=" * 50 + "\n\n"

            # Overview
            overview = report['overview']
            report_text += f"Total Rows: {overview['total_rows']:,}\n"
            report_text += f"Total Columns: {overview['total_columns']}\n"
            report_text += f"Memory Usage: {overview['memory_usage_mb']:.2f} MB\n\n"

            # Missing data
            missing = report['missing_data']
            report_text += f"Missing Values: {missing['total_missing']:,} ({missing['missing_percentage']:.2f}%)\n\n"

            if missing['columns_with_missing']:
                report_text += "Columns with Missing Data:\n"
                for col, info in missing['columns_with_missing'].items():
                    report_text += f"  {col}: {info['count']} ({info['percentage']:.1f}%)\n"
                report_text += "\n"

            # Duplicates
            duplicates = report['duplicates']
            report_text += f"Duplicate Rows: {duplicates['duplicate_rows']} ({duplicates['duplicate_percentage']:.2f}%)\n\n"

            self.preprocessing_results.delete(1.0, tk.END)
            self.preprocessing_results.insert(1.0, report_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate quality report: {str(e)}")

    def auto_clean_data(self):
        """Automatically clean data"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            # Define auto-cleaning pipeline
            transformations = {
                'clean_names': {},
                'handle_missing': {'strategy': 'auto'},
                'remove_duplicates': {},
                'convert_dtypes': {}
            }

            # Apply transformations
            cleaned_data = self.data_processor.apply_data_transformations(
                self.current_data, transformations)

            # Update current data
            rows_before = len(self.current_data)
            self.current_data = cleaned_data
            rows_after = len(self.current_data)

            # Register with analytics engine
            self.analytics_engine.register_dataframe(self.current_data)

            # Update display
            self.update_data_display()

            # Show results
            result_text = f"AUTO-CLEANING RESULTS\n{'=' * 30}\n\n"
            result_text += f"Rows before: {rows_before:,}\n"
            result_text += f"Rows after: {rows_after:,}\n"
            result_text += f"Rows removed: {rows_before - rows_after:,}\n\n"
            result_text += "Applied transformations:\n"
            result_text += "• Cleaned column names\n"
            result_text += "• Handled missing values\n"
            result_text += "• Removed duplicate rows\n"
            result_text += "• Converted data types\n"

            self.preprocessing_results.delete(1.0, tk.END)
            self.preprocessing_results.insert(1.0, result_text)

            self.status_var.set("Data cleaned successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-clean data: {str(e)}")

    def handle_missing_values(self):
        """Handle missing values"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            strategy = self.missing_strategy_var.get()

            before_count = self.current_data.isnull().sum().sum()
            self.current_data = self.data_processor.handle_missing_values(
                self.current_data, strategy)
            after_count = self.current_data.isnull().sum().sum()

            # Register with analytics engine
            self.analytics_engine.register_dataframe(self.current_data)

            # Update display
            self.update_data_display()

            result_text = f"MISSING VALUES HANDLING\n{'=' * 30}\n\n"
            result_text += f"Strategy: {strategy}\n"
            result_text += f"Missing values before: {before_count:,}\n"
            result_text += f"Missing values after: {after_count:,}\n"
            result_text += f"Missing values removed: {before_count - after_count:,}\n"

            self.preprocessing_results.delete(1.0, tk.END)
            self.preprocessing_results.insert(1.0, result_text)

            self.status_var.set("Missing values handled successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle missing values: {str(e)}")

    def remove_duplicates(self):
        """Remove duplicate rows"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            rows_before = len(self.current_data)
            self.current_data = self.data_processor.remove_duplicates(self.current_data)
            rows_after = len(self.current_data)

            # Register with analytics engine
            self.analytics_engine.register_dataframe(self.current_data)

            # Update display
            self.update_data_display()

            result_text = f"DUPLICATE REMOVAL\n{'=' * 20}\n\n"
            result_text += f"Rows before: {rows_before:,}\n"
            result_text += f"Rows after: {rows_after:,}\n"
            result_text += f"Duplicates removed: {rows_before - rows_after:,}\n"

            self.preprocessing_results.delete(1.0, tk.END)
            self.preprocessing_results.insert(1.0, result_text)

            self.status_var.set("Duplicates removed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove duplicates: {str(e)}")

    def handle_outliers(self):
        """Handle outliers"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            method = self.outlier_method_var.get()

            rows_before = len(self.current_data)
            self.current_data = self.data_processor.handle_outliers(
                self.current_data, method=method, action='remove')
            rows_after = len(self.current_data)

            # Register with analytics engine
            self.analytics_engine.register_dataframe(self.current_data)

            # Update display
            self.update_data_display()

            result_text = f"OUTLIER HANDLING\n{'=' * 20}\n\n"
            result_text += f"Method: {method}\n"
            result_text += f"Rows before: {rows_before:,}\n"
            result_text += f"Rows after: {rows_after:,}\n"
            result_text += f"Outliers removed: {rows_before - rows_after:,}\n"

            self.preprocessing_results.delete(1.0, tk.END)
            self.preprocessing_results.insert(1.0, result_text)

            self.status_var.set("Outliers handled successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle outliers: {str(e)}")

    # =============================================================================
    # Analytics Methods
    # =============================================================================

    def on_analysis_type_change(self, event=None):
        """Handle analysis type change"""
        analysis_type = self.analysis_type_var.get()

        # Show/hide column selection for certain analyses
        if analysis_type in ["Group Analysis", "Association Analysis"]:
            self.column_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10), after=self.notebook)
            self.create_column_selection()
        else:
            self.column_frame.grid_remove()

    def create_column_selection(self):
        """Create column selection widgets"""
        # Clear existing widgets
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        if self.current_data is None:
            return

        columns = list(self.current_data.columns)

        ttk.Label(self.column_frame, text="Select Columns:").grid(row=0, column=0, sticky=tk.W)

        listbox_frame = ttk.Frame(self.column_frame)
        listbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.column_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=6)
        self.column_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.column_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.column_listbox.configure(yscrollcommand=scrollbar.set)

        for col in columns:
            self.column_listbox.insert(tk.END, col)

    def run_analysis(self):
        """Run selected analysis"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        analysis_type = self.analysis_type_var.get()
        self.status_var.set(f"Running {analysis_type}...")
        self.root.update()

        try:
            if analysis_type == "Summary Statistics":
                result = self.analytics_engine.generate_summary_statistics(self.current_data)
                result_text = self.format_summary_statistics(result)

            elif analysis_type == "Correlation Analysis":
                result = self.analytics_engine.generate_correlation_analysis(self.current_data)
                result_text = self.format_correlation_analysis(result)

            elif analysis_type == "Group Analysis":
                selected_cols = self.get_selected_columns()
                if not selected_cols:
                    messagebox.showerror("Error", "Please select columns for group analysis")
                    return
                result = self.analytics_engine.generate_group_analysis(self.current_data, selected_cols)
                result_text = self.format_group_analysis(result)

            elif analysis_type == "Time Series Analysis":
                # Find date columns
                date_cols = [col for col in self.current_data.columns
                             if pd.api.types.is_datetime64_any_dtype(self.current_data[col])]
                if not date_cols:
                    messagebox.showerror("Error", "No datetime columns found")
                    return
                result = self.analytics_engine.generate_time_series_analysis(
                    self.current_data, date_cols[0])
                result_text = self.format_time_series_analysis(result)

            elif analysis_type == "Distribution Analysis":
                result = self.analytics_engine.generate_distribution_analysis(self.current_data)
                result_text = self.format_distribution_analysis(result)

            elif analysis_type == "Association Analysis":
                target_col = self.target_column_var.get()
                if not target_col:
                    messagebox.showerror("Error", "Please select target column")
                    return
                result = self.analytics_engine.generate_association_analysis(
                    self.current_data, target_col)
                result_text = self.format_association_analysis(result)

            else:
                result_text = "Analysis type not implemented"

            # Store in history
            self.analysis_history.append({
                'type': analysis_type,
                'timestamp': datetime.now(),
                'result': result if 'result' in locals() else {}
            })

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, result_text)
            self.status_var.set("Analysis completed")

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.status_var.set("Analysis failed")

    def get_selected_columns(self):
        """Get selected columns from listbox"""
        if hasattr(self, 'column_listbox'):
            selected_indices = self.column_listbox.curselection()
            return [self.column_listbox.get(i) for i in selected_indices]
        return []

    def format_summary_statistics(self, result):
        """Format summary statistics for display"""
        text = "SUMMARY STATISTICS\n" + "=" * 50 + "\n\n"

        overview = result['dataset_overview']
        text += f"Total Rows: {overview['total_rows']:,}\n"
        text += f"Total Columns: {overview['total_columns']}\n"
        text += f"Memory Usage: {overview['memory_usage_mb']:.2f} MB\n"
        text += f"Missing Values: {overview['missing_values_total']:,} ({overview['missing_percentage']:.2f}%)\n\n"

        # Numeric summary
        if result['numeric_summary']:
            text += "NUMERIC COLUMNS\n" + "-" * 20 + "\n"
            for col, stats in result['numeric_summary'].items():
                text += f"\n{col}:\n"
                text += f"  Count: {stats['count']}\n"
                text += f"  Mean: {stats['mean']:.3f}\n"
                text += f"  Std: {stats['std']:.3f}\n"
                text += f"  Min: {stats['min']:.3f}\n"
                text += f"  Max: {stats['max']:.3f}\n"

        return text

    def format_correlation_analysis(self, result):
        """Format correlation analysis for display"""
        text = "CORRELATION ANALYSIS\n" + "=" * 50 + "\n\n"
        text += f"Method: {result['method']}\n\n"

        if result['significant_correlations']:
            text += "TOP CORRELATIONS:\n" + "-" * 20 + "\n"
            for corr in result['significant_correlations'][:10]:
                text += f"{corr['variable1']} - {corr['variable2']}: {corr['correlation']:.3f}\n"

        return text

    def format_group_analysis(self, result):
        """Format group analysis for display"""
        text = "GROUP ANALYSIS\n" + "=" * 50 + "\n\n"

        for group_col, analysis in result.items():
            text += f"GROUPING BY: {group_col}\n" + "-" * 30 + "\n"

            counts = analysis['group_counts']
            for group, count in list(counts.items())[:10]:
                text += f"{group}: {count}\n"
            text += "\n"

        return text

    def format_time_series_analysis(self, result):
        """Format time series analysis for display"""
        text = "TIME SERIES ANALYSIS\n" + "=" * 50 + "\n\n"

        date_range = result['date_range']
        text += f"Date Range: {date_range['start_date']} to {date_range['end_date']}\n"
        text += f"Total Days: {date_range['total_days']}\n"
        text += f"Total Records: {date_range['total_records']}\n\n"

        return text

    def format_distribution_analysis(self, result):
        """Format distribution analysis for display"""
        text = "DISTRIBUTION ANALYSIS\n" + "=" * 50 + "\n\n"

        for col, stats in result.items():
            text += f"{col}:\n"
            normality = stats['normality_test']
            text += f"  Normal Distribution: {'Yes' if normality['is_normal'] else 'No'}\n"
            text += f"  Skewness: {stats['distribution_stats']['skewness']:.3f}\n"
            text += f"  Kurtosis: {stats['distribution_stats']['kurtosis']:.3f}\n\n"

        return text

    def format_association_analysis(self, result):
        """Format association analysis for display"""
        text = "ASSOCIATION ANALYSIS\n" + "=" * 50 + "\n\n"
        text += f"Target Variable: {result['target_info']['column']}\n\n"

        if result['numeric_associations']:
            text += "NUMERIC ASSOCIATIONS:\n" + "-" * 25 + "\n"
            for var, stats in result['numeric_associations'].items():
                text += f"{var}: r={stats['pearson_correlation']:.3f} ({stats['strength']})\n"

        return text

    # =============================================================================
    # Chart Methods
    # =============================================================================

    def on_chart_type_change(self, event=None):
        """Handle chart type change"""
        chart_type = self.chart_type_var.get()

        # Enable/disable column selections based on chart type
        if chart_type in ["histogram", "box", "violin", "distribution"]:
            self.y_column_combo.config(state='disabled')
        else:
            self.y_column_combo.config(state='normal')

        if chart_type == "correlation_heatmap":
            self.x_column_combo.config(state='disabled')
            self.y_column_combo.config(state='disabled')
        else:
            self.x_column_combo.config(state='normal')

    def generate_chart(self):
        """Generate selected chart"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        chart_type = self.chart_type_var.get()
        x_col = self.x_column_var.get()
        y_col = self.y_column_var.get()

        if not x_col and chart_type not in ["correlation_heatmap"]:
            messagebox.showerror("Error", "Please select X column")
            return

        self.status_var.set(f"Generating {chart_type}...")
        self.root.update()

        try:
            # Clear previous chart
            self.figure.clear()

            # Generate chart using chart generator
            if chart_type == "histogram":
                self.figure = self.chart_generator.create_histogram(self.current_data, x_col)
            elif chart_type == "bar":
                self.figure = self.chart_generator.create_bar_chart(self.current_data, x_col, y_col)
            elif chart_type == "line":
                if not y_col:
                    messagebox.showerror("Error", "Please select Y column for line chart")
                    return
                self.figure = self.chart_generator.create_line_chart(self.current_data, x_col, y_col)
            elif chart_type == "scatter":
                if not y_col:
                    messagebox.showerror("Error", "Please select Y column for scatter plot")
                    return
                self.figure = self.chart_generator.create_scatter_plot(self.current_data, x_col, y_col)
            elif chart_type == "box":
                self.figure = self.chart_generator.create_box_plot(self.current_data, x_col)
            elif chart_type == "violin":
                self.figure = self.chart_generator.create_violin_plot(self.current_data, x_col)
            elif chart_type == "correlation_heatmap":
                self.figure = self.chart_generator.create_correlation_heatmap(self.current_data)
            elif chart_type == "distribution":
                self.figure = self.chart_generator.create_distribution_plot(self.current_data, x_col)

            # Update canvas with new figure
            self.chart_canvas.figure = self.figure
            self.chart_canvas.draw()

            # Store chart
            self.current_chart_id += 1
            self.charts[self.current_chart_id] = self.figure

            self.status_var.set("Chart generated successfully")

        except Exception as e:
            messagebox.showerror("Chart Error", str(e))
            self.status_var.set("Chart generation failed")

    def save_chart(self):
        """Save current chart"""
        if not hasattr(self, 'figure') or not self.figure.get_axes():
            messagebox.showerror("Error", "No chart to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Chart",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                self.chart_generator.save_chart(self.figure, file_path)
                messagebox.showinfo("Success", f"Chart saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {e}")

    def clear_chart(self):
        """Clear current chart"""
        self.figure.clear()
        self.chart_canvas.draw()

    # =============================================================================
    # Machine Learning Methods
    # =============================================================================

    def run_ml_task(self):
        """Run machine learning task"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        task_type = self.ml_task_var.get()
        target_column = self.target_column_var.get()

        if task_type in ["Classification", "Regression"] and not target_column:
            messagebox.showerror("Error", "Please select target column")
            return

        self.status_var.set(f"Running {task_type}...")
        self.root.update()

        try:
            if task_type == "Classification":
                result = self.ml_engine.train_classification_model(self.current_data, target_column)
                result_text = self.format_ml_classification_result(result)

            elif task_type == "Regression":
                result = self.ml_engine.train_regression_model(self.current_data, target_column)
                result_text = self.format_ml_regression_result(result)

            elif task_type == "Clustering":
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    messagebox.showerror("Error", "No numeric columns for clustering")
                    return
                result = self.ml_engine.perform_clustering(self.current_data, numeric_cols)
                result_text = self.format_ml_clustering_result(result)

            elif task_type == "PCA":
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    messagebox.showerror("Error", "No numeric columns for PCA")
                    return
                result = self.ml_engine.perform_pca(self.current_data, numeric_cols)
                result_text = self.format_ml_pca_result(result)

            elif task_type == "Auto ML":
                if pd.api.types.is_numeric_dtype(self.current_data[target_column]):
                    result = self.ml_engine.auto_ml_regression(self.current_data, target_column)
                    result_text = self.format_auto_ml_result(result, "regression")
                else:
                    result = self.ml_engine.auto_ml_classification(self.current_data, target_column)
                    result_text = self.format_auto_ml_result(result, "classification")

            else:
                result_text = "ML task not implemented"

            # Display results
            self.ml_results_text.delete(1.0, tk.END)
            self.ml_results_text.insert(1.0, result_text)
            self.status_var.set("ML task completed")

        except Exception as e:
            messagebox.showerror("ML Error", str(e))
            self.status_var.set("ML task failed")

    def format_ml_classification_result(self, result):
        """Format classification results"""
        text = "CLASSIFICATION RESULTS\n" + "=" * 50 + "\n\n"
        text += f"Model: {result['model_name']}\n"
        text += f"Accuracy: {result['accuracy']:.4f}\n"
        text += f"Precision: {result['precision']:.4f}\n"
        text += f"Recall: {result['recall']:.4f}\n"
        text += f"F1 Score: {result['f1_score']:.4f}\n"
        text += f"Cross-validation Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n\n"

        if result['feature_importance']:
            text += "TOP FEATURE IMPORTANCE:\n" + "-" * 25 + "\n"
            for feature, importance in list(result['feature_importance'].items())[:10]:
                text += f"{feature}: {importance:.4f}\n"

        return text

    def format_ml_regression_result(self, result):
        """Format regression results"""
        text = "REGRESSION RESULTS\n" + "=" * 50 + "\n\n"
        text += f"Model: {result['model_name']}\n"
        text += f"Mean Squared Error: {result['mse']:.4f}\n"
        text += f"Mean Absolute Error: {result['mae']:.4f}\n"
        text += f"R² Score: {result['r2_score']:.4f}\n"
        text += f"RMSE: {result['rmse']:.4f}\n"
        text += f"Cross-validation Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n\n"

        if result['feature_importance']:
            text += "TOP FEATURE IMPORTANCE:\n" + "-" * 25 + "\n"
            for feature, importance in list(result['feature_importance'].items())[:10]:
                text += f"{feature}: {importance:.4f}\n"

        return text

    def format_ml_clustering_result(self, result):
        """Format clustering results"""
        text = "CLUSTERING RESULTS\n" + "=" * 50 + "\n\n"
        text += f"Algorithm: {result['algorithm']}\n"
        text += f"Number of Clusters: {result['n_clusters']}\n"
        text += f"Silhouette Score: {result['silhouette_score']:.4f}\n\n"

        text += "CLUSTER SUMMARY:\n" + "-" * 20 + "\n"
        for cluster, info in result['cluster_summary'].items():
            text += f"{cluster}: {info['size']} samples ({info['percentage']:.1f}%)\n"

        return text

    def format_ml_pca_result(self, result):
        """Format PCA results"""
        text = "PCA RESULTS\n" + "=" * 50 + "\n\n"
        text += f"Number of Components: {result['n_components']}\n"
        text += f"Total Variance Explained: {result['total_variance_explained']:.4f}\n\n"

        text += "EXPLAINED VARIANCE BY COMPONENT:\n" + "-" * 35 + "\n"
        for i, var in enumerate(result['explained_variance_ratio'][:10]):
            text += f"PC{i + 1}: {var:.4f}\n"

        return text

    def format_auto_ml_result(self, result, task_type):
        """Format Auto ML results"""
        text = f"AUTO ML RESULTS ({task_type.upper()})\n" + "=" * 50 + "\n\n"
        text += f"Best Model: {result['best_model']}\n"

        if task_type == "classification":
            text += f"Best Accuracy: {result['best_score']:.4f}\n\n"
        else:
            text += f"Best MSE: {result['best_score']:.4f}\n\n"

        text += "MODEL COMPARISON:\n" + "-" * 20 + "\n"
        for model, score in result['model_comparison'].items():
            text += f"{model}: {score:.4f}\n"

        return text

    # =============================================================================
    # Export Methods
    # =============================================================================

    def export_data(self):
        """Export current data"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data to export")
            return

        export_format = self.export_format_var.get()

        # File dialog
        if export_format == "csv":
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
        elif export_format == "excel":
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
        elif export_format == "json":
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
        elif export_format == "parquet":
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".parquet",
                filetypes=[("Parquet files", "*.parquet")]
            )
        elif export_format == "html":
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html")]
            )
        else:
            messagebox.showerror("Error", "Unsupported export format")
            return

        if not file_path:
            return

        try:
            self.status_var.set("Exporting data...")
            self.root.update()

            success = self.export_manager.export_data(self.current_data, file_path, export_format)

            if success:
                # Add to export history
                self.export_history.append({
                    'filepath': file_path,
                    'format': export_format,
                    'timestamp': datetime.now(),
                    'rows': len(self.current_data),
                    'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
                })

                messagebox.showinfo("Success", f"Data exported to {file_path}")
                self.status_var.set("Export completed")
            else:
                messagebox.showerror("Error", "Export failed")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self.status_var.set("Export failed")

    def generate_summary_report(self):
        """Generate summary report"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            report = self.export_manager.generate_summary_report(
                self.current_data, self.data_info)

            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)

            # Save report
            file_path = filedialog.asksaveasfilename(
                title="Save Summary Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Report Error", str(e))

    def generate_analysis_report(self):
        """Generate analysis report"""
        if not self.analysis_history:
            messagebox.showerror("Error", "No analysis results to report")
            return

        try:
            # Collect all analysis results
            analysis_results = {}
            for analysis in self.analysis_history:
                analysis_results[analysis['type']] = analysis['result']

            report = self.export_manager.generate_analysis_report(
                analysis_results, self.current_data)

            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)

            # Save report
            file_path = filedialog.asksaveasfilename(
                title="Save Analysis Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Report Error", str(e))

    def generate_dashboard(self):
        """Generate interactive dashboard"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            # Collect charts
            chart_figures = list(self.charts.values()) if self.charts else []

            # Collect analysis results
            analysis_results = {}
            for analysis in self.analysis_history:
                analysis_results[analysis['type']] = analysis['result']

            # Save dashboard
            file_path = filedialog.asksaveasfilename(
                title="Save Dashboard",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html")]
            )

            if file_path:
                success = self.export_manager.export_dashboard(
                    self.current_data, chart_figures, analysis_results, file_path)

                if success:
                    messagebox.showinfo("Success", f"Dashboard saved to {file_path}")
                    # Ask if user wants to open it
                    if messagebox.askyesno("Open Dashboard", "Would you like to open the dashboard in your browser?"):
                        webbrowser.open(f"file://{Path(file_path).absolute()}")

        except Exception as e:
            messagebox.showerror("Dashboard Error", str(e))

    # =============================================================================
    # Database Methods
    # =============================================================================

    def show_database_dialog(self):
        """Show database connection dialog"""
        DatabaseDialog(self.root, self)

    def show_query_dialog(self):
        """Show SQL query dialog"""
        QueryDialog(self.root, self)

    def list_database_tables(self):
        """List database tables"""
        try:
            tables = self.database_manager.get_table_list()

            info_text = "DATABASE TABLES\n" + "=" * 30 + "\n\n"
            if tables:
                for i, table in enumerate(tables, 1):
                    info_text += f"{i}. {table}\n"
            else:
                info_text += "No tables found or no active connection"

            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(1.0, info_text)

        except Exception as e:
            messagebox.showerror("Database Error", str(e))

    # =============================================================================
    # Settings Methods
    # =============================================================================

    def change_theme(self, event=None):
        """Change application theme"""
        theme = self.theme_var.get()
        style = ttk.Style()
        style.theme_use(theme)

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def on_closing(self):
        """Handle application closing"""
        try:
            # Close database connections
            self.database_manager.close_all_connections()

            # Ask for confirmation
            if messagebox.askokcancel("Quit", "Do you want to quit InSpectra Analytics?"):
                self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")
            self.root.destroy()


# =============================================================================
# Dialog Classes
# =============================================================================

class DatabaseDialog:
    """Database connection dialog"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Database Connection")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Database type
        ttk.Label(main_frame, text="Database Type:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self.db_type_var = tk.StringVar(value="sqlite")
        db_combo = ttk.Combobox(main_frame, textvariable=self.db_type_var,
                                values=["sqlite", "duckdb", "postgresql", "mysql", "mssql"])
        db_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))

        # Connection fields
        fields = [
            ("Host:", "host", "localhost"),
            ("Port:", "port", "5432"),
            ("Database:", "database", ""),
            ("Username:", "username", ""),
            ("Password:", "password", "")
        ]

        self.field_vars = {}
        for i, (label, key, default) in enumerate(fields, 1):
            ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=(0, 10))
            var = tk.StringVar(value=default)
            self.field_vars[key] = var

            if key == "password":
                entry = ttk.Entry(main_frame, textvariable=var, show="*")
            else:
                entry = ttk.Entry(main_frame, textvariable=var)

            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="Test Connection",
                   command=self.test_connection).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Connect",
                   command=self.connect).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel",
                   command=self.dialog.destroy).grid(row=0, column=2)

        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)

    def test_connection(self):
        """Test database connection"""
        try:
            db_type = self.db_type_var.get()
            connection_params = {key: var.get() for key, var in self.field_vars.items()}

            success = self.main_app.database_manager.test_connection(db_type, connection_params)

            if success:
                messagebox.showinfo("Success", "Connection test successful!")
            else:
                messagebox.showerror("Error", "Connection test failed")

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def connect(self):
        """Connect to database"""
        try:
            db_type = self.db_type_var.get()
            connection_params = {key: var.get() for key, var in self.field_vars.items()}

            connection_name = self.main_app.database_manager.connect_database(
                db_type, connection_params)

            messagebox.showinfo("Success", f"Connected as '{connection_name}'")
            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))


class QueryDialog:
    """SQL query dialog"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("SQL Query")
        self.dialog.geometry("700x500")

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Query input
        ttk.Label(main_frame, text="SQL Query:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        self.query_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15)
        self.query_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Sample query
        sample_query = """-- Sample queries
SELECT * FROM table_name LIMIT 100;

-- Column statistics
SELECT COUNT(*) as total_rows FROM table_name;

-- Add your query here..."""

        self.query_text.insert(1.0, sample_query)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="Execute",
                   command=self.execute_query).grid(row=0, column=0)
        ttk.Button(button_frame, text="Clear",
                   command=self.clear_query).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(button_frame, text="Close",
                   command=self.dialog.destroy).grid(row=0, column=2, padx=(10, 0))

        # Configure dialog grid
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)

    def execute_query(self):
        """Execute SQL query"""
        query = self.query_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showerror("Error", "Please enter a query")
            return

        try:
            result_df = self.main_app.database_manager.execute_query(query)

            # Load result into main app
            self.main_app.current_data = result_df
            self.main_app.data_info = {
                'filename': 'SQL Query Result',
                'rows': len(result_df),
                'columns': len(result_df.columns),
                'memory': f"{result_df.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB",
                'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Register with analytics engine
            self.main_app.analytics_engine.register_dataframe(result_df)

            # Update display
            self.main_app.update_data_display()

            messagebox.showinfo("Success", f"Query executed successfully. Loaded {len(result_df)} rows.")
            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Query Error", str(e))

    def clear_query(self):
        """Clear query text"""
        self.query_text.delete(1.0, tk.END)


# =============================================================================
# Main Application Runner
# =============================================================================

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = InSpectraGUI(root)

    # Start the application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", str(e))


if __name__ == "__main__":
    main()