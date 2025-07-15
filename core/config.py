# config.py
# Application configuration settings

import os
from pathlib import Path

# Application Information
APP_NAME = "InSpectra Analytics Platform"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Advanced Data Processing & Analytics Suite"
APP_AUTHOR = "InSpectra Analytics Team"

# File Paths and Directories
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
EXPORTS_DIR = APP_DIR / "exports"
REPORTS_DIR = APP_DIR / "reports"
TEMP_DIR = APP_DIR / "temp"
LOGS_DIR = APP_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, EXPORTS_DIR, REPORTS_DIR, TEMP_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# GUI Settings
DEFAULT_WINDOW_SIZE = (1600, 1000)
MIN_WINDOW_SIZE = (1400, 900)
DEFAULT_THEME = "clam"
AVAILABLE_THEMES = ["clam", "alt", "default", "classic"]

# Data Processing Settings
MAX_PREVIEW_ROWS = 10000
DEFAULT_PREVIEW_ROWS = 1000
MAX_MEMORY_USAGE_MB = 2048  # 2GB default limit
CHUNK_SIZE = 10000  # For large file processing

# Chart Settings
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
CHART_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Export Settings
DEFAULT_EXPORT_FORMAT = "csv"
SUPPORTED_EXPORT_FORMATS = ["csv", "excel", "json", "parquet", "html", "xml"]
DEFAULT_CHART_FORMAT = "png"
SUPPORTED_CHART_FORMATS = ["png", "pdf", "svg", "jpg", "eps"]

# Database Settings
DEFAULT_DB_TYPE = "sqlite"
SUPPORTED_DB_TYPES = ["sqlite", "duckdb", "postgresql", "mysql", "mssql"]
DEFAULT_DB_TIMEOUT = 30  # seconds
MAX_DB_CONNECTIONS = 10

# Machine Learning Settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_STATE = 42
MAX_FEATURES_FOR_AUTO_ML = 50

# Analytics Settings
DEFAULT_CORRELATION_METHOD = "pearson"
MIN_CORRELATION_THRESHOLD = 0.1
OUTLIER_METHODS = ["iqr", "zscore"]
MISSING_VALUE_STRATEGIES = ["auto", "drop_rows", "drop_columns", "mean", "median", "mode"]

# File Format Settings
CSV_ENCODING = "utf-8"
CSV_DELIMITERS = [",", ";", "\t", "|"]
EXCEL_ENGINES = ["openpyxl", "xlrd"]
JSON_ORIENTATIONS = ["records", "index", "values", "split", "table"]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "inspectra.log"
MAX_LOG_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

# Performance Settings
ENABLE_PERFORMANCE_MONITORING = True
MEMORY_WARNING_THRESHOLD = 0.8  # 80% of available memory
PROCESSING_TIMEOUT = 300  # 5 minutes

# UI Customization
COLORS = {
    'primary': '#3498db',
    'secondary': '#2c3e50',
    'success': '#27ae60',
    'warning': '#f39c12',
    'error': '#e74c3c',
    'info': '#3498db',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

FONTS = {
    'default': ('Arial', 10),
    'title': ('Arial', 18, 'bold'),
    'header': ('Arial', 12, 'bold'),
    'small': ('Arial', 8)
}

# Advanced Settings
ENABLE_MULTITHREADING = True
MAX_WORKER_THREADS = 4
ENABLE_PROGRESS_BARS = True
AUTO_SAVE_INTERVAL = 300  # seconds (5 minutes)

# Sample Data Configuration
SAMPLE_DATASETS = {
    'iris': {
        'name': 'Iris Dataset',
        'description': 'Classic flower classification dataset',
        'features': 4,
        'samples': 150,
        'task_type': 'classification'
    },
    'titanic': {
        'name': 'Titanic Dataset',
        'description': 'Passenger survival analysis',
        'features': 9,
        'samples': 891,
        'task_type': 'classification'
    },
    'boston housing': {
        'name': 'Boston Housing',
        'description': 'Real estate price prediction',
        'features': 13,
        'samples': 506,
        'task_type': 'regression'
    },
    'wine quality': {
        'name': 'Wine Quality',
        'description': 'Wine quality assessment',
        'features': 11,
        'samples': 1599,
        'task_type': 'regression'
    },
    'stock market': {
        'name': 'Stock Market',
        'description': 'Time series financial data',
        'features': 7,
        'samples': 1461,
        'task_type': 'time_series'
    }
}

# Chart Type Configurations
CHART_CONFIGS = {
    'histogram': {
        'requires_y': False,
        'data_types': ['numeric'],
        'description': 'Distribution of numeric values'
    },
    'bar': {
        'requires_y': False,
        'data_types': ['categorical', 'numeric'],
        'description': 'Comparison of categories'
    },
    'line': {
        'requires_y': True,
        'data_types': ['numeric', 'datetime'],
        'description': 'Trends over time or continuous data'
    },
    'scatter': {
        'requires_y': True,
        'data_types': ['numeric'],
        'description': 'Relationship between two numeric variables'
    },
    'box': {
        'requires_y': False,
        'data_types': ['numeric'],
        'description': 'Distribution with quartiles and outliers'
    },
    'violin': {
        'requires_y': False,
        'data_types': ['numeric'],
        'description': 'Distribution shape and density'
    },
    'correlation_heatmap': {
        'requires_y': False,
        'data_types': ['numeric'],
        'description': 'Correlation matrix visualization'
    },
    'distribution': {
        'requires_y': False,
        'data_types': ['numeric'],
        'description': 'Histogram with density curve'
    }
}

# ML Model Configurations
ML_MODELS = {
    'classification': {
        'Random Forest': {
            'class': 'RandomForestClassifier',
            'params': {'n_estimators': 100, 'random_state': DEFAULT_RANDOM_STATE}
        },
        'Logistic Regression': {
            'class': 'LogisticRegression',
            'params': {'random_state': DEFAULT_RANDOM_STATE}
        },
        'SVM': {
            'class': 'SVC',
            'params': {'random_state': DEFAULT_RANDOM_STATE}
        },
        'Decision Tree': {
            'class': 'DecisionTreeClassifier',
            'params': {'random_state': DEFAULT_RANDOM_STATE}
        },
        'Naive Bayes': {
            'class': 'GaussianNB',
            'params': {}
        },
        'K-Nearest Neighbors': {
            'class': 'KNeighborsClassifier',
            'params': {}
        }
    },
    'regression': {
        'Random Forest': {
            'class': 'RandomForestRegressor',
            'params': {'n_estimators': 100, 'random_state': DEFAULT_RANDOM_STATE}
        },
        'Linear Regression': {
            'class': 'LinearRegression',
            'params': {}
        },
        'SVM': {
            'class': 'SVR',
            'params': {}
        },
        'Decision Tree': {
            'class': 'DecisionTreeRegressor',
            'params': {'random_state': DEFAULT_RANDOM_STATE}
        },
        'K-Nearest Neighbors': {
            'class': 'KNeighborsRegressor',
            'params': {}
        }
    },
    'clustering': {
        'K-Means': {
            'class': 'KMeans',
            'params': {'n_clusters': 3, 'random_state': DEFAULT_RANDOM_STATE}
        },
        'DBSCAN': {
            'class': 'DBSCAN',
            'params': {}
        },
        'Agglomerative': {
            'class': 'AgglomerativeClustering',
            'params': {'n_clusters': 3}
        }
    }
}

# Feature Engineering Settings
FEATURE_ENGINEERING = {
    'polynomial_degree': 2,
    'max_bins': 10,
    'datetime_features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'],
    'text_features': ['length', 'word_count', 'char_count']
}

# Data Quality Thresholds
DATA_QUALITY = {
    'max_missing_percentage': 50,  # Drop columns with >50% missing
    'min_unique_values': 2,  # Minimum unique values for analysis
    'max_unique_ratio': 0.95,  # Maximum unique ratio for categorical
    'outlier_threshold': 0.05,  # 5% outlier threshold
    'correlation_threshold': 0.95  # High correlation threshold
}

# Report Templates
REPORT_TEMPLATES = {
    'summary': {
        'sections': ['overview', 'columns', 'quality', 'recommendations'],
        'include_charts': True
    },
    'analysis': {
        'sections': ['summary', 'correlations', 'distributions', 'outliers'],
        'include_charts': True
    },
    'ml_report': {
        'sections': ['data_prep', 'model_performance', 'feature_importance'],
        'include_charts': True
    }
}

# API Settings (for future web interface)
API_SETTINGS = {
    'host': 'localhost',
    'port': 8000,
    'debug': False,
    'max_upload_size': 100 * 1024 * 1024,  # 100MB
    'timeout': 30
}

# Security Settings
SECURITY = {
    'sanitize_inputs': True,
    'validate_file_types': True,
    'max_query_length': 10000,
    'allowed_file_extensions': ['.csv', '.xlsx', '.xls', '.parquet', '.json', '.txt'],
    'blocked_sql_keywords': ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE USER']
}


# Environment-specific Settings
def get_env_config():
    """Get environment-specific configuration"""
    env = os.getenv('INSPECTRA_ENV', 'development')

    if env == 'production':
        return {
            'LOG_LEVEL': 'WARNING',
            'DEBUG': False,
            'ENABLE_PERFORMANCE_MONITORING': True,
            'MAX_MEMORY_USAGE_MB': 4096
        }
    elif env == 'testing':
        return {
            'LOG_LEVEL': 'DEBUG',
            'DEBUG': True,
            'MAX_PREVIEW_ROWS': 100,
            'PROCESSING_TIMEOUT': 10
        }
    else:  # development
        return {
            'LOG_LEVEL': 'DEBUG',
            'DEBUG': True,
            'ENABLE_PERFORMANCE_MONITORING': False
        }


# Apply environment-specific settings
ENV_CONFIG = get_env_config()
locals().update(ENV_CONFIG)

# Version and Update Settings
VERSION_INFO = {
    'major': 2,
    'minor': 0,
    'patch': 0,
    'build': '20240101',
    'full_version': f"{APP_VERSION}",
    'release_date': '2024-01-01'
}

UPDATE_SETTINGS = {
    'check_for_updates': True,
    'update_url': 'https://api.inspectra-analytics.com/updates',
    'auto_update': False
}

# Help and Documentation
HELP_URLS = {
    'documentation': 'https://docs.inspectra-analytics.com',
    'tutorials': 'https://docs.inspectra-analytics.com/tutorials',
    'api_reference': 'https://docs.inspectra-analytics.com/api',
    'support': 'https://support.inspectra-analytics.com',
    'github': 'https://github.com/inspectra/analytics'
}

# Experimental Features (can be toggled)
EXPERIMENTAL_FEATURES = {
    'advanced_ml_models': False,
    'real_time_data': False,
    'cloud_integration': False,
    'collaborative_features': False,
    'web_interface': False
}

# Default User Preferences (can be customized)
DEFAULT_PREFERENCES = {
    'theme': DEFAULT_THEME,
    'auto_save': True,
    'show_tooltips': True,
    'remember_window_size': True,
    'default_chart_type': 'histogram',
    'confirm_destructive_actions': True,
    'show_welcome_screen': True
}