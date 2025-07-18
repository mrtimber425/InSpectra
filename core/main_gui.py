# main_gui.py
# Complete InSpectra Analytics GUI Application with AI Analysis

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
from ai_analysis_engine import AIAnalysisEngine  # New AI analysis engine
from export_manager import ExportManager
from database_manager import DatabaseManager
from filter_engine import FilterEngine  # New filter engine

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')


class InSpectraGUI:
    """Main InSpectra Analytics GUI Application with AI Analysis"""

    def __init__(self, root):
        self.root = root
        self.root.title("InSpectra Analytics - AI-Powered Data Analysis Platform")
        self.root.geometry("1800x1200")  # Increased window size
        self.root.minsize(1600, 1000)
        self.running = True
        self.check_results_id = None
        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.analytics_engine = AnalyticsEngine()
        self.chart_generator = ChartGenerator()
        self.ai_analysis_engine = AIAnalysisEngine()  # New AI analysis engine
        self.export_manager = ExportManager()
        self.database_manager = DatabaseManager()
        self.filter_engine = FilterEngine()  # New filter engine

        # Initialize data storage
        self.current_data = None
        self.original_data = None
        self.filtered_data = None  # For storing filtered data
        self.active_filters = []  # Track active filters
        self.data_info = {}
        self.result_queue = queue.Queue()
        self.analysis_history = []
        self.export_history = []

        # Chart storage
        self.charts = {}
        self.current_chart_id = 0

        # AI Analysis storage
        self.loaded_models = {}
        self.ai_analysis_results = []

        # Setup styles and create GUI
        self.setup_styles()
        self.create_widgets()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Start background tasks
        self.check_results_id = None  # Initialize to None
        self.check_results_id = self.root.after(100, self.check_results)
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

        # Enhanced data table style
        style.configure("Enhanced.Treeview",
                        font=('Consolas', 10),
                        rowheight=25)
        style.configure("Enhanced.Treeview.Heading",
                        font=('Arial', 11, 'bold'),
                        background='#34495e',
                        foreground='white')

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
        self.create_data_view_tab()  # New Data View tab
        self.create_preprocessing_tab()
        self.create_analytics_tab()
        self.create_charts_tab()
        self.create_ai_analysis_tab()  # New AI Analysis tab
        self.create_export_tab()
        self.create_database_tab()
        self.create_settings_tab()

        # Status bar
        self.create_status_bar(main_frame)

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

    def create_title_section(self, parent):
        """Create title section"""
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        title_frame.grid_columnconfigure(0, weight=1)

        title_label = ttk.Label(title_frame, text="🚀 InSpectra Analytics Platform v2.0",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0)

        subtitle_label = ttk.Label(title_frame, text="Advanced Data Processing & AI-Powered Analytics Suite",
                                   style='Info.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(2, 0))

    def create_data_tab(self):
        """Create data management tab (without preview - moved to Data View tab)"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📁 Data Manager")

        data_frame.grid_rowconfigure(2, weight=1)
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

        # Data operations (basic)
        ops_frame = ttk.LabelFrame(data_frame, text="Data Operations", padding="15")
        ops_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(ops_frame, text="📋 Copy to Clipboard",
                   command=self.copy_to_clipboard).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(ops_frame, text="🧹 Reset Data",
                   command=self.reset_data).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(ops_frame, text="ℹ️ Data Info",
                   command=self.show_data_info).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(ops_frame, text="👁️ View Data",
                   command=lambda: self.notebook.select(1)).grid(row=0, column=3,
                                                                 padx=(0, 5))  # Switch to Data View tab

    def create_data_view_tab(self):
        """Create comprehensive data view tab with filtering"""
        view_frame = ttk.Frame(self.notebook)
        self.notebook.add(view_frame, text="👁️ Data View")

        view_frame.grid_rowconfigure(3, weight=1)
        view_frame.grid_columnconfigure(0, weight=1)

        # Filter controls section
        self.create_filter_controls(view_frame)

        # Data statistics bar
        self.create_data_stats_bar(view_frame)

        # Active filters display (initially hidden)
        self.create_active_filters_display(view_frame)

        # Enhanced data table with filtering
        self.create_filtered_data_table(view_frame)

    def create_filter_controls(self, parent):
        """Create filter controls section"""
        filter_frame = ttk.LabelFrame(parent, text="Data Filters", padding="15")
        filter_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        filter_frame.grid_columnconfigure(1, weight=1)
        filter_frame.grid_columnconfigure(3, weight=1)
        filter_frame.grid_columnconfigure(5, weight=1)

        # Column selection for filtering
        ttk.Label(filter_frame, text="Filter Column:").grid(row=0, column=0, sticky=tk.W)
        self.filter_column_var = tk.StringVar()
        self.filter_column_combo = ttk.Combobox(filter_frame, textvariable=self.filter_column_var, width=15)
        self.filter_column_combo.grid(row=0, column=1, padx=(5, 10), sticky=(tk.W, tk.E))
        self.filter_column_combo.bind('<<ComboboxSelected>>', self.on_filter_column_change)

        # Filter type selection
        ttk.Label(filter_frame, text="Filter Type:").grid(row=0, column=2, sticky=tk.W)
        self.filter_type_var = tk.StringVar(value="contains")
        self.filter_type_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type_var, width=12)
        self.filter_type_combo['values'] = ["contains", "equals", "not_equals", "starts_with",
                                            "ends_with", "greater_than", "less_than", "between", "is_null", "not_null"]
        self.filter_type_combo.grid(row=0, column=3, padx=(5, 10), sticky=(tk.W, tk.E))
        self.filter_type_combo.bind('<<ComboboxSelected>>', self.on_filter_type_change)

        # Filter value entry
        ttk.Label(filter_frame, text="Filter Value:").grid(row=0, column=4, sticky=tk.W)
        self.filter_value_var = tk.StringVar()
        self.filter_value_entry = ttk.Entry(filter_frame, textvariable=self.filter_value_var, width=15)
        self.filter_value_entry.grid(row=0, column=5, padx=(5, 10), sticky=(tk.W, tk.E))
        self.filter_value_entry.bind('<Return>', lambda e: self.apply_filter())

        # Second value for "between" filter (initially hidden)
        self.filter_value2_var = tk.StringVar()
        self.filter_value2_entry = ttk.Entry(filter_frame, textvariable=self.filter_value2_var, width=10)
        self.filter_value2_label = ttk.Label(filter_frame, text="to:")

        # Filter buttons
        button_frame = ttk.Frame(filter_frame)
        button_frame.grid(row=0, column=6, padx=(10, 0))

        ttk.Button(button_frame, text="Apply", command=self.apply_filter).grid(row=0, column=0)
        ttk.Button(button_frame, text="Clear", command=self.clear_filters).grid(row=0, column=1, padx=(5, 0))

        # Advanced filters section
        advanced_frame = ttk.Frame(filter_frame)
        advanced_frame.grid(row=1, column=0, columnspan=7, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(advanced_frame, text="Quick Filters:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)

        ttk.Button(advanced_frame, text="Show Nulls Only",
                   command=lambda: self.quick_filter("nulls")).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(advanced_frame, text="Hide Nulls",
                   command=lambda: self.quick_filter("no_nulls")).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(advanced_frame, text="Show Duplicates",
                   command=lambda: self.quick_filter("duplicates")).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(advanced_frame, text="Unique Values Only",
                   command=lambda: self.quick_filter("unique")).grid(row=0, column=4, padx=(0, 5))

    def create_data_stats_bar(self, parent):
        """Create data statistics bar"""
        stats_frame = ttk.Frame(parent)
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.grid_columnconfigure(0, weight=1)

        # Statistics labels
        stats_info_frame = ttk.Frame(stats_frame)
        stats_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.filtered_stats_label = ttk.Label(stats_info_frame, text="", style='Info.TLabel')
        self.filtered_stats_label.grid(row=0, column=0, sticky=tk.W)

        # Export filtered data button
        export_frame = ttk.Frame(stats_frame)
        export_frame.grid(row=0, column=1, sticky=tk.E)

        ttk.Button(export_frame, text="📤 Export Filtered Data",
                   command=self.export_filtered_data).grid(row=0, column=0)
        ttk.Button(export_frame, text="🔄 Refresh View",
                   command=self.refresh_filtered_view).grid(row=0, column=1, padx=(5, 0))

    def create_active_filters_display(self, parent):
        """Create active filters display"""
        self.active_filters_frame = ttk.LabelFrame(parent, text="Active Filters", padding="10")
        # This will be shown only when filters are applied
        self.active_filters_scrolled = scrolledtext.ScrolledText(self.active_filters_frame,
                                                                 wrap=tk.WORD, height=3)

    def create_filtered_data_table(self, parent):
        """Create the main data table with filtering support"""
        table_frame = ttk.LabelFrame(parent, text="Data Preview", padding="15")
        table_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Table controls
        controls_frame = ttk.Frame(table_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(controls_frame, text="Rows to display:").grid(row=0, column=0, sticky=tk.W)
        self.preview_rows_var = tk.StringVar(value="1000")
        rows_entry = ttk.Entry(controls_frame, textvariable=self.preview_rows_var, width=10)
        rows_entry.grid(row=0, column=1, padx=(5, 0))

        ttk.Button(controls_frame, text="Refresh",
                   command=self.refresh_filtered_view).grid(row=0, column=2, padx=(10, 0))

        # Create Treeview with custom styling
        data_table_frame = ttk.Frame(table_frame)
        data_table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        data_table_frame.grid_rowconfigure(0, weight=1)
        data_table_frame.grid_columnconfigure(0, weight=1)

        self.filtered_data_tree = ttk.Treeview(data_table_frame, show='headings', style="Enhanced.Treeview")
        self.filtered_data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Enhanced scrollbars
        v_scrollbar = ttk.Scrollbar(data_table_frame, orient=tk.VERTICAL, command=self.filtered_data_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.filtered_data_tree.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(data_table_frame, orient=tk.HORIZONTAL, command=self.filtered_data_tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.filtered_data_tree.configure(xscrollcommand=h_scrollbar.set)

        # Add context menu for data table
        self.create_filtered_data_context_menu()

    def create_filtered_data_context_menu(self):
        """Create context menu for filtered data table"""
        self.filtered_data_context_menu = tk.Menu(self.root, tearoff=0)
        self.filtered_data_context_menu.add_command(label="Copy Cell", command=self.copy_selected_cell_filtered)
        self.filtered_data_context_menu.add_command(label="Copy Row", command=self.copy_selected_row_filtered)
        self.filtered_data_context_menu.add_command(label="Copy Column", command=self.copy_selected_column_filtered)
        self.filtered_data_context_menu.add_separator()
        self.filtered_data_context_menu.add_command(label="Show Column Stats", command=self.show_column_stats_filtered)
        self.filtered_data_context_menu.add_command(label="Filter by This Value", command=self.filter_by_selected_value)
        self.filtered_data_context_menu.add_command(label="Exclude This Value", command=self.exclude_selected_value)

        self.filtered_data_tree.bind("<Button-3>", self.show_filtered_data_context_menu)

    # =============================================================================
    # Filter Methods
    # =============================================================================

    def on_filter_column_change(self, event=None):
        """Handle filter column selection change"""
        column = self.filter_column_var.get()
        if not column or self.current_data is None:
            return

        # Update filter type options based on column data type
        if column in self.current_data.columns:
            dtype = self.current_data[column].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                filter_types = ["equals", "not_equals", "greater_than", "less_than", "between", "is_null", "not_null"]
            else:
                filter_types = ["contains", "equals", "not_equals", "starts_with", "ends_with", "is_null", "not_null"]

            self.filter_type_combo['values'] = filter_types
            if self.filter_type_var.get() not in filter_types:
                self.filter_type_var.set(filter_types[0])

    def on_filter_type_change(self, event=None):
        """Handle filter type change"""
        filter_type = self.filter_type_var.get()

        # Show/hide second value entry for "between" filter
        if filter_type == "between":
            self.filter_value2_label.grid(row=0, column=7, sticky=tk.W, padx=(5, 0))
            self.filter_value2_entry.grid(row=0, column=8, padx=(5, 0))
        else:
            self.filter_value2_label.grid_remove()
            self.filter_value2_entry.grid_remove()

        # Disable value entry for null/not_null filters
        if filter_type in ["is_null", "not_null"]:
            self.filter_value_entry.config(state='disabled')
        else:
            self.filter_value_entry.config(state='normal')

    def apply_filter(self):
        """Apply the current filter"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        column = self.filter_column_var.get()
        filter_type = self.filter_type_var.get()
        value = self.filter_value_var.get()
        value2 = self.filter_value2_var.get()

        if not column:
            messagebox.showerror("Error", "Please select a column to filter")
            return

        if filter_type not in ["is_null", "not_null"] and not value:
            messagebox.showerror("Error", "Please enter a filter value")
            return

        try:
            # Create filter object
            filter_obj = {
                'column': column,
                'type': filter_type,
                'value': value,
                'value2': value2 if filter_type == "between" else None
            }

            # Apply filter using filter engine
            if self.filtered_data is None:
                self.filtered_data = self.current_data.copy()

            self.filtered_data = self.filter_engine.apply_filter(self.filtered_data, filter_obj)

            # Add to active filters
            self.active_filters.append(filter_obj)

            # Update display
            self.refresh_filtered_view()
            self.update_active_filters_display()

            self.status_var.set(f"Filter applied. Showing {len(self.filtered_data)} of {len(self.current_data)} rows")

        except Exception as e:
            messagebox.showerror("Filter Error", f"Failed to apply filter: {str(e)}")

    def quick_filter(self, filter_type):
        """Apply quick filters"""
        if self.current_data is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            if filter_type == "nulls":
                # Show only rows with null values
                self.filtered_data = self.current_data[self.current_data.isnull().any(axis=1)]
                filter_desc = "Showing rows with null values"

            elif filter_type == "no_nulls":
                # Hide rows with null values
                self.filtered_data = self.current_data.dropna()
                filter_desc = "Hiding rows with null values"

            elif filter_type == "duplicates":
                # Show only duplicate rows
                self.filtered_data = self.current_data[self.current_data.duplicated(keep=False)]
                filter_desc = "Showing duplicate rows"

            elif filter_type == "unique":
                # Show only unique rows
                self.filtered_data = self.current_data.drop_duplicates()
                filter_desc = "Showing unique rows only"

            # Add to active filters
            self.active_filters.append({
                'type': 'quick_filter',
                'filter_type': filter_type,
                'description': filter_desc
            })

            # Update display
            self.refresh_filtered_view()
            self.update_active_filters_display()

            self.status_var.set(f"{filter_desc}. Showing {len(self.filtered_data)} of {len(self.current_data)} rows")

        except Exception as e:
            messagebox.showerror("Filter Error", f"Failed to apply quick filter: {str(e)}")

    def clear_filters(self):
        """Clear all active filters"""
        self.filtered_data = None
        self.active_filters = []

        # Clear filter inputs
        self.filter_column_var.set("")
        self.filter_type_var.set("contains")
        self.filter_value_var.set("")
        self.filter_value2_var.set("")

        # Hide active filters display
        self.active_filters_frame.grid_remove()

        # Refresh view
        self.refresh_filtered_view()

        if self.current_data is not None:
            self.status_var.set(f"Filters cleared. Showing all {len(self.current_data)} rows")

    def update_active_filters_display(self):
        """Update the active filters display"""
        if not self.active_filters:
            self.active_filters_frame.grid_remove()
            return

        # Show active filters frame
        self.active_filters_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.active_filters_frame.grid_rowconfigure(0, weight=1)
        self.active_filters_frame.grid_columnconfigure(0, weight=1)

        if not hasattr(self, 'active_filters_scrolled'):
            self.active_filters_scrolled = scrolledtext.ScrolledText(self.active_filters_frame,
                                                                     wrap=tk.WORD, height=3)

        self.active_filters_scrolled.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Clear and update content
        self.active_filters_scrolled.delete(1.0, tk.END)

        filter_text = "Active Filters:\n"
        for i, f in enumerate(self.active_filters, 1):
            if f.get('type') == 'quick_filter':
                filter_text += f"{i}. {f['description']}\n"
            else:
                filter_text += f"{i}. {f['column']} {f['type']} '{f['value']}'"
                if f.get('value2'):
                    filter_text += f" and '{f['value2']}'"
                filter_text += "\n"

        self.active_filters_scrolled.insert(1.0, filter_text)

    def refresh_filtered_view(self):
        """Refresh the filtered data view"""
        # Determine which data to display
        display_data = self.filtered_data if self.filtered_data is not None else self.current_data

        if display_data is None:
            return

        try:
            max_rows = int(self.preview_rows_var.get())
        except ValueError:
            max_rows = 1000

        # Clear existing data
        self.filtered_data_tree.delete(*self.filtered_data_tree.get_children())

        # Set up columns
        columns = list(display_data.columns)
        self.filtered_data_tree["columns"] = columns

        for col in columns:
            self.filtered_data_tree.heading(col, text=col, anchor='center')
            max_width = max(len(col) * 10, 150)
            self.filtered_data_tree.column(col, width=max_width, anchor='center')

        # Add data rows
        preview_data = display_data.head(max_rows)
        for index, row in preview_data.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("NULL")
                elif isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))

            item = self.filtered_data_tree.insert("", "end", values=values)
            # Add tags for alternating row colors
            if index % 2 == 0:
                self.filtered_data_tree.item(item, tags=('evenrow',))
            else:
                self.filtered_data_tree.item(item, tags=('oddrow',))

        # Configure row colors
        self.filtered_data_tree.tag_configure('evenrow', background='#f8f9fa')
        self.filtered_data_tree.tag_configure('oddrow', background='white')

        # Update statistics
        self.update_filtered_stats(display_data, max_rows)

    def update_filtered_stats(self, display_data, max_rows):
        """Update filtered data statistics"""
        if display_data is None:
            return

        total_rows = len(display_data)
        displaying = min(max_rows, total_rows)

        if self.current_data is not None:
            original_rows = len(self.current_data)
            if self.filtered_data is not None:
                stats_text = f"Showing {displaying:,} of {total_rows:,} filtered rows "
                stats_text += f"(from {original_rows:,} total) | "
            else:
                stats_text = f"Showing {displaying:,} of {total_rows:,} rows | "
        else:
            stats_text = f"Showing {displaying:,} of {total_rows:,} rows | "

        stats_text += f"{len(display_data.columns)} columns | "
        stats_text += f"{display_data.isnull().sum().sum():,} null values"

        self.filtered_stats_label.config(text=stats_text)

    def export_filtered_data(self):
        """Export currently filtered data"""
        export_data = self.filtered_data if self.filtered_data is not None else self.current_data

        if export_data is None:
            messagebox.showerror("Error", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Filtered Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    export_data.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    export_data.to_excel(file_path, index=False)
                else:
                    export_data.to_csv(file_path, index=False)

                filter_info = f" ({len(self.active_filters)} filters applied)" if self.active_filters else ""
                messagebox.showinfo("Success", f"Filtered data exported to {file_path}{filter_info}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export filtered data: {str(e)}")

    # Context menu methods for filtered data
    def show_filtered_data_context_menu(self, event):
        """Show context menu for filtered data table"""
        try:
            self.filtered_data_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.filtered_data_context_menu.grab_release()

    def copy_selected_cell_filtered(self):
        """Copy selected cell value to clipboard"""
        selection = self.filtered_data_tree.selection()
        if selection:
            item = selection[0]
            values = self.filtered_data_tree.item(item)['values']
            if values:
                self.root.clipboard_clear()
                self.root.clipboard_append(str(values[0]))

    def copy_selected_row_filtered(self):
        """Copy selected row to clipboard"""
        selection = self.filtered_data_tree.selection()
        if selection:
            item = selection[0]
            values = self.filtered_data_tree.item(item)['values']
            if values:
                row_text = '\t'.join(str(val) for val in values)
                self.root.clipboard_clear()
                self.root.clipboard_append(row_text)

    def copy_selected_column_filtered(self):
        """Copy selected column to clipboard"""
        messagebox.showinfo("Info", "Column copy feature coming soon!")

    def show_column_stats_filtered(self):
        """Show statistics for selected column"""
        messagebox.showinfo("Info", "Column statistics feature coming soon!")

    def filter_by_selected_value(self):
        """Filter by the selected cell value"""
        selection = self.filtered_data_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a cell first")
            return

        item = selection[0]
        values = self.filtered_data_tree.item(item)['values']
        if values and self.current_data is not None:
            # Get column index
            columns = list(self.filtered_data_tree["columns"])
            # For simplicity, assume first column - in a full implementation,
            # you'd determine which column was clicked
            messagebox.showinfo("Info", f"Filter by value '{values[0]}' feature coming soon!")

    def exclude_selected_value(self):
        """Exclude the selected cell value"""
        selection = self.filtered_data_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a cell first")
            return

        messagebox.showinfo("Info", "Exclude value feature coming soon!")

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
            self.filtered_data = None  # Clear any previous filters
            self.active_filters = []  # Clear active filters
            self.data_info = info

            # Register with analytics engine
            self.analytics_engine.register_dataframe(df)

            self.progress_var.set(100)
            self.result_queue.put(("success", "File loaded successfully"))

        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def load_sample_data(self, dataset_name):
        """Load sample dataset"""
        try:
            self.status_var.set(f"Loading {dataset_name} dataset...")
            df, info = self.data_loader.get_sample_data(dataset_name)

            self.current_data = df
            self.original_data = df.copy()
            self.filtered_data = None  # Clear any previous filters
            self.active_filters = []  # Clear active filters
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
                df = self.data_loader.load_from_url(url)

                info = {
                    "filename": "URL Data",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory": f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB",
                    "load_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "file_size": "N/A"
                }

                self.current_data = df
                self.original_data = df.copy()
                self.filtered_data = None  # Clear any previous filters
                self.active_filters = []  # Clear active filters
                self.data_info = info

                # Register with analytics engine
                self.analytics_engine.register_dataframe(df)

                self.update_data_display()
                self.status_var.set("URL data loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load from URL: {str(e)}")

    def update_data_display(self):
        """Update data display after loading"""
        if self.current_data is None:
            return

        # Update info labels
        for key, label in self.info_labels.items():
            if key in self.data_info:
                label.config(text=str(self.data_info[key]))

        # Update column selections for filters
        self.update_filter_column_selections()

        # Update column selections for charts
        self.update_column_selections()

        # Refresh filtered view (will show all data since no filters applied)
        self.refresh_filtered_view()

    def update_filter_column_selections(self):
        """Update filter column combo box"""
        if self.current_data is None:
            return

        columns = list(self.current_data.columns)
        if hasattr(self, 'filter_column_combo'):
            self.filter_column_combo['values'] = columns

    def refresh_preview(self):
        """Enhanced refresh data preview table with statistics"""
        # This method is kept for backward compatibility but now redirects to filtered view
        self.refresh_filtered_view()

    def update_column_selections(self):
        """Update column combo boxes"""
        if self.current_data is None:
            return

        columns = list(self.current_data.columns)

        # Update chart column selections
        if hasattr(self, 'x_column_combo'):
            self.x_column_combo['values'] = columns
            self.y_column_combo['values'] = columns

        if columns:
            if hasattr(self, 'x_column_var'):
                self.x_column_var.set(columns[0])
            if len(columns) > 1 and hasattr(self, 'y_column_var'):
                self.y_column_var.set(columns[1])

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
        self.filtered_data = None  # Clear filters
        self.active_filters = []  # Clear active filters
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
                # Use the last column as target by default
                target_col = self.current_data.columns[-1]
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
    # AI Analysis Methods
    # =============================================================================

    def create_ai_analysis_tab(self):
        """Create AI Analysis tab with Hugging Face model support"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI Analysis")

        ai_frame.grid_rowconfigure(2, weight=1)
        ai_frame.grid_columnconfigure(0, weight=1)

        # Model Management Section
        model_frame = ttk.LabelFrame(ai_frame, text="AI Model Management", padding="15")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Hugging Face model input
        hf_frame = ttk.Frame(model_frame)
        hf_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        hf_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(hf_frame, text="🤗 Hugging Face Model:").grid(row=0, column=0, sticky=tk.W)
        self.hf_model_var = tk.StringVar()
        hf_entry = ttk.Entry(hf_frame, textvariable=self.hf_model_var, width=50)
        hf_entry.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        ttk.Button(hf_frame, text="Load HF Model",
                   command=self.load_huggingface_model).grid(row=0, column=2, padx=(5, 0))

        # Popular models suggestions
        suggestions_frame = ttk.Frame(model_frame)
        suggestions_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(suggestions_frame, text="Popular Models:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))

        popular_models = [
            ("📊 Data Analysis", "microsoft/DialoGPT-large"),
            ("📈 Financial Analysis", "ProsusAI/finbert"),
            ("🔍 Text Classification", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            ("📝 Text Generation", "gpt2"),
        ]

        for i, (name, model_id) in enumerate(popular_models):
            ttk.Button(suggestions_frame, text=name,
                       command=lambda m=model_id: self.load_suggested_model(m)).grid(
                row=(i // 2) + 1, column=i % 2, padx=(0, 5), pady=(0, 5), sticky=tk.W)

        # Local model file loading
        local_frame = ttk.Frame(model_frame)
        local_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        local_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(local_frame, text="💾 Local Model File:").grid(row=0, column=0, sticky=tk.W)
        self.local_model_var = tk.StringVar()
        local_entry = ttk.Entry(local_frame, textvariable=self.local_model_var, width=50)
        local_entry.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        ttk.Button(local_frame, text="Browse",
                   command=self.browse_local_model).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(local_frame, text="Load Local",
                   command=self.load_local_model).grid(row=0, column=3, padx=(5, 0))

        # Analysis Controls Section
        analysis_frame = ttk.LabelFrame(ai_frame, text="AI Analysis Controls", padding="15")
        analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Analysis type selection
        analysis_type_frame = ttk.Frame(analysis_frame)
        analysis_type_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(analysis_type_frame, text="Analysis Type:").grid(row=0, column=0, sticky=tk.W)
        self.ai_analysis_type_var = tk.StringVar(value="Data Insights")
        analysis_combo = ttk.Combobox(analysis_type_frame, textvariable=self.ai_analysis_type_var,
                                      values=["Data Insights", "Pattern Detection", "Anomaly Detection",
                                              "Trend Analysis", "Predictive Analysis", "Custom Prompt"])
        analysis_combo.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        # Model selection
        ttk.Label(analysis_type_frame, text="Model:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.selected_model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(analysis_type_frame, textvariable=self.selected_model_var)
        self.model_combo.grid(row=0, column=3, padx=(5, 0))

        # Custom prompt area
        prompt_frame = ttk.Frame(analysis_frame)
        prompt_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        prompt_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(prompt_frame, text="Custom Analysis Prompt:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W)

        self.custom_prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=4)
        self.custom_prompt_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # Analysis buttons
        button_frame = ttk.Frame(analysis_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(button_frame, text="🚀 Run AI Analysis",
                   command=self.run_ai_analysis).grid(row=0, column=0)
        ttk.Button(button_frame, text="💾 Save Analysis",
                   command=self.save_ai_analysis).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(button_frame, text="📋 Analysis History",
                   command=self.show_analysis_history).grid(row=0, column=2, padx=(10, 0))

        # Results Section
        results_frame = ttk.LabelFrame(ai_frame, text="AI Analysis Results", padding="15")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # Create notebook for different result types
        self.ai_results_notebook = ttk.Notebook(results_frame)
        self.ai_results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Text results tab
        text_results_frame = ttk.Frame(self.ai_results_notebook)
        self.ai_results_notebook.add(text_results_frame, text="📝 Text Results")
        text_results_frame.grid_rowconfigure(0, weight=1)
        text_results_frame.grid_columnconfigure(0, weight=1)

        self.ai_results_text = scrolledtext.ScrolledText(text_results_frame, wrap=tk.WORD, height=20)
        self.ai_results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Visual results tab
        visual_results_frame = ttk.Frame(self.ai_results_notebook)
        self.ai_results_notebook.add(visual_results_frame, text="📊 Visual Results")
        visual_results_frame.grid_rowconfigure(0, weight=1)
        visual_results_frame.grid_columnconfigure(0, weight=1)

        # Matplotlib figure for AI-generated visualizations
        self.ai_figure = Figure(figsize=(12, 8), dpi=100)
        self.ai_chart_canvas = FigureCanvasTkAgg(self.ai_figure, visual_results_frame)
        self.ai_chart_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_huggingface_model(self):
        """Load model from Hugging Face Hub"""
        model_name = self.hf_model_var.get().strip()
        if not model_name:
            messagebox.showerror("Error", "Please enter a Hugging Face model name")
            return

        self.status_var.set(f"Loading model: {model_name}...")
        self.root.update()

        threading.Thread(target=self._load_hf_model_thread, args=(model_name,), daemon=True).start()

    def _load_hf_model_thread(self, model_name):
        """Load Hugging Face model in background thread"""
        try:
            success = self.ai_analysis_engine.load_huggingface_model(model_name)
            if success:
                self.result_queue.put(("model_loaded", model_name))
                # Update model dropdown
                current_models = list(self.ai_analysis_engine.get_loaded_models().keys())
                self.root.after(100, lambda: self.update_model_dropdown(current_models))
            else:
                self.result_queue.put(("error", f"Failed to load model: {model_name}"))
        except Exception as e:
            self.result_queue.put(("error", f"Error loading model: {str(e)}"))

    def load_suggested_model(self, model_id):
        """Load a suggested model"""
        self.hf_model_var.set(model_id)
        self.load_huggingface_model()

    def browse_local_model(self):
        """Browse for local model file"""
        file_path = filedialog.askdirectory(title="Select Model Directory")
        if file_path:
            self.local_model_var.set(file_path)

    def load_local_model(self):
        """Load model from local file"""
        model_path = self.local_model_var.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please select a model directory")
            return

        try:
            success = self.ai_analysis_engine.load_local_model(model_path)
            if success:
                model_name = Path(model_path).name
                messagebox.showinfo("Success", f"Local model '{model_name}' loaded successfully")
                # Update model dropdown
                current_models = list(self.ai_analysis_engine.get_loaded_models().keys())
                self.update_model_dropdown(current_models)
            else:
                messagebox.showerror("Error", "Failed to load local model")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading local model: {str(e)}")

    def update_model_dropdown(self, models):
        """Update the model selection dropdown"""
        self.model_combo['values'] = models
        if models:
            self.selected_model_var.set(models[0])

    def run_ai_analysis(self):
        """Run AI analysis on current data"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        model_name = self.selected_model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please load and select a model first")
            return

        analysis_type = self.ai_analysis_type_var.get()
        custom_prompt = self.custom_prompt_text.get(1.0, tk.END).strip()

        self.status_var.set(f"Running AI analysis with {model_name}...")
        self.root.update()

        threading.Thread(target=self._run_ai_analysis_thread,
                         args=(model_name, analysis_type, custom_prompt), daemon=True).start()

    def _run_ai_analysis_thread(self, model_name, analysis_type, custom_prompt):
        """Run AI analysis in background thread"""
        try:
            result = self.ai_analysis_engine.analyze_data(
                self.current_data,
                model_name,
                analysis_type,
                custom_prompt if custom_prompt else None
            )

            self.ai_analysis_results.append({
                'timestamp': datetime.now(),
                'model': model_name,
                'analysis_type': analysis_type,
                'result': result
            })

            self.result_queue.put(("ai_analysis_complete", result))
        except Exception as e:
            self.result_queue.put(("error", f"AI analysis failed: {str(e)}"))

    def save_ai_analysis(self):
        """Save current AI analysis results"""
        if not self.ai_analysis_results:
            messagebox.showerror("Error", "No analysis results to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save AI Analysis Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.ai_analysis_results, f, indent=2, default=str)
                else:
                    # Save as text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for result in self.ai_analysis_results:
                            f.write(f"Analysis Date: {result['timestamp']}\n")
                            f.write(f"Model: {result['model']}\n")
                            f.write(f"Type: {result['analysis_type']}\n")
                            f.write(f"Results:\n{result['result']}\n")
                            f.write("-" * 80 + "\n\n")

                messagebox.showinfo("Success", f"Analysis results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def show_analysis_history(self):
        """Show AI analysis history in a popup window"""
        if not self.ai_analysis_results:
            messagebox.showinfo("Info", "No analysis history available")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("AI Analysis History")
        history_window.geometry("800x600")

        history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD)
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        history_content = "AI ANALYSIS HISTORY\n" + "=" * 50 + "\n\n"
        for i, result in enumerate(self.ai_analysis_results, 1):
            history_content += f"Analysis #{i}\n"
            history_content += f"Date: {result['timestamp']}\n"
            history_content += f"Model: {result['model']}\n"
            history_content += f"Type: {result['analysis_type']}\n"
            history_content += f"Results Preview: {str(result['result'])[:200]}...\n"
            history_content += "-" * 40 + "\n\n"

        history_text.insert(tk.END, history_content)

    # =============================================================================
    # Export Methods
    # =============================================================================

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

        # AI settings
        ai_settings = ttk.LabelFrame(settings_frame, text="AI Analysis Settings", padding="15")
        ai_settings.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.ai_auto_download_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ai_settings, text="Auto-download Hugging Face models",
                        variable=self.ai_auto_download_var).grid(row=0, column=0, sticky=tk.W)

        self.ai_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ai_settings, text="Use GPU acceleration (if available)",
                        variable=self.ai_gpu_var).grid(row=1, column=0, sticky=tk.W)

        # About section
        about_frame = ttk.LabelFrame(settings_frame, text="About", padding="15")
        about_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        about_text = """
InSpectra Analytics Platform v2.0 with AI Analysis

A comprehensive data analytics and visualization tool built with Python, now featuring
AI-powered analysis capabilities using Hugging Face models.

Features:
• Advanced data loading and preprocessing
• Statistical analysis and machine learning
• Interactive data visualization
• AI-powered analysis with custom models
• Database connectivity
• Report generation and export

Built with: Tkinter, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Transformers
        """

        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT)
        about_label.grid(row=0, column=0, sticky=(tk.W, tk.N))

    def change_theme(self, event=None):
        """Change application theme"""
        theme = self.theme_var.get()
        style = ttk.Style()
        style.theme_use(theme)

    # =============================================================================
    # Background Processing Methods
    # =============================================================================

    def check_results(self):
        """Enhanced check for results from background threads"""
        if not self.running:
            return

        try:
            while True:
                result_type, *args = self.result_queue.get_nowait()

                if result_type == "success":
                    self.status_var.set(args[0])
                    self.update_data_display()
                elif result_type == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Error", args[0])
                elif result_type == "model_loaded":
                    self.status_var.set(f"Model {args[0]} loaded successfully")
                    messagebox.showinfo("Success", f"Model '{args[0]}' loaded successfully")
                elif result_type == "ai_analysis_complete":
                    self.status_var.set("AI analysis completed")
                    self.display_ai_results(args[0])

                self.progress_var.set(0)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in check_results: {e}")
            self.running = False
            return

        # Only schedule next check if still running and window exists
        if self.running:
            try:
                if self.root.winfo_exists():
                    self.check_results_id = self.root.after(100, self.check_results)
                else:
                    self.running = False
            except (tk.TclError, AttributeError):
                # Window was destroyed, stop the loop
                self.running = False

    def display_ai_results(self, results):
        """Display AI analysis results"""
        try:
            # Display text results
            if isinstance(results, dict):
                formatted_results = json.dumps(results, indent=2)
            else:
                formatted_results = str(results)

            self.ai_results_text.delete(1.0, tk.END)
            self.ai_results_text.insert(1.0, formatted_results)

            # Switch to results tab
            self.ai_results_notebook.select(0)  # Select text results tab

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def on_closing(self):
        """Enhanced cleanup when closing application"""
        try:
            # Stop the check_results loop first
            self.running = False

            # Cancel any pending after() calls
            if hasattr(self, 'check_results_id') and self.check_results_id:
                try:
                    self.root.after_cancel(self.check_results_id)
                    self.check_results_id = None
                except:
                    pass

            # Close database connections
            if hasattr(self, 'database_manager'):
                self.database_manager.close_all_connections()

            # Cleanup AI models
            if hasattr(self, 'ai_analysis_engine'):
                self.ai_analysis_engine.cleanup()

            # Destroy the window first, then quit
            self.root.destroy()

        except Exception as e:
            print(f"Error during closing: {e}")
            # Force exit even if cleanup fails
            try:
                self.root.destroy()
            except:
                pass
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
            self.main_app.filtered_data = None  # Clear any filters
            self.main_app.active_filters = []  # Clear active filters
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


class SearchDialog:
    """Data search dialog"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Search Data")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        """Create search dialog widgets"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Search options
        ttk.Label(main_frame, text="Search in:").grid(row=0, column=0, sticky=tk.W)
        self.search_column_var = tk.StringVar()
        self.search_column_combo = ttk.Combobox(main_frame, textvariable=self.search_column_var)

        if self.main_app.current_data is not None:
            columns = ['All Columns'] + list(self.main_app.current_data.columns)
            self.search_column_combo['values'] = columns
            self.search_column_var.set('All Columns')

        self.search_column_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

        # Search term
        ttk.Label(main_frame, text="Search for:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.search_term_var = tk.StringVar()
        search_entry = ttk.Entry(main_frame, textvariable=self.search_term_var)
        search_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(10, 0))

        # Search options
        self.case_sensitive_var = tk.BooleanVar()
        ttk.Checkbutton(main_frame, text="Case sensitive",
                        variable=self.case_sensitive_var).grid(row=2, column=0, columnspan=2,
                                                               sticky=tk.W, pady=(10, 0))

        self.regex_var = tk.BooleanVar()
        ttk.Checkbutton(main_frame, text="Regular expression",
                        variable=self.regex_var).grid(row=3, column=0, columnspan=2,
                                                      sticky=tk.W, pady=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="Search",
                   command=self.perform_search).grid(row=0, column=0)
        ttk.Button(button_frame, text="Cancel",
                   command=self.dialog.destroy).grid(row=0, column=1, padx=(10, 0))

        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)

    def perform_search(self):
        """Perform the search"""
        search_term = self.search_term_var.get().strip()
        if not search_term:
            messagebox.showerror("Error", "Please enter a search term")
            return

        try:
            # Implement search logic here
            column = self.search_column_var.get()
            case_sensitive = self.case_sensitive_var.get()
            is_regex = self.regex_var.get()

            if self.main_app.current_data is None:
                messagebox.showerror("Error", "No data loaded")
                return

            df = self.main_app.current_data

            if column == "All Columns":
                # Search in all columns
                if is_regex:
                    mask = df.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=case_sensitive, regex=True, na=False)).any(axis=1)
                else:
                    if case_sensitive:
                        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=True, na=False)).any(
                            axis=1)
                    else:
                        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(
                            axis=1)
            else:
                # Search in specific column
                if column not in df.columns:
                    messagebox.showerror("Error", f"Column '{column}' not found")
                    return

                if is_regex:
                    mask = df[column].astype(str).str.contains(search_term, case=case_sensitive, regex=True, na=False)
                else:
                    mask = df[column].astype(str).str.contains(search_term, case=case_sensitive, na=False)

            # Apply search as filter
            search_results = df[mask]

            if len(search_results) == 0:
                messagebox.showinfo("Search Results", f"No results found for '{search_term}'")
            else:
                # Apply search results as filter
                self.main_app.filtered_data = search_results

                # Add to active filters
                search_filter = {
                    'type': 'search',
                    'column': column,
                    'search_term': search_term,
                    'case_sensitive': case_sensitive,
                    'regex': is_regex,
                    'description': f"Search for '{search_term}' in {column}"
                }
                self.main_app.active_filters.append(search_filter)

                # Update display
                self.main_app.refresh_filtered_view()
                self.main_app.update_active_filters_display()

                messagebox.showinfo("Search Results",
                                    f"Found {len(search_results)} rows containing '{search_term}'")

                # Switch to Data View tab to show results
                self.main_app.notebook.select(1)

            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Search Error", f"Search failed: {str(e)}")


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