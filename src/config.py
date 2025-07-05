"""
Inspectra Configuration Management
Centralized configuration for the advanced forensic analysis platform
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

class InspectraConfig:
    """
    Centralized configuration management for Inspectra
    Handles application settings, detection thresholds, and user preferences
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self._config = {
            # Application Settings
            "application": {
                "name": "Inspectra",
                "version": "3.0.0",
                "description": "Advanced Forensic Analysis Platform",
                "author": "Inspectra Development Team",
                "website": "https://inspectra.local",
                "support_email": "support@inspectra.local"
            },

            # User Interface Settings
            "ui": {
                "theme": "dark",
                "default_window_size": [1600, 1000],
                "minimum_window_size": [1400, 900],
                "table_row_height": 40,
                "table_min_column_width": 120,
                "chart_figure_size": [12, 8],
                "chart_dpi": 100,
                "auto_save_layout": True,
                "show_splash_screen": True,
                "enable_animations": True,
                "font_family": "Segoe UI",
                "font_size": 12
            },

            # Analysis Settings
            "analysis": {
                "default_confidence_threshold": 0.7,
                "max_anomalies_display": 1000,
                "enable_ml_enhancement": True,
                "auto_feature_engineering": True,
                "parallel_processing": True,
                "cache_analysis_results": True,
                "result_cache_hours": 24
            },

            # Network Forensics Settings
            "network": {
                "port_scan_threshold": 10,
                "ddos_threshold": 1000,
                "data_exfil_threshold": 104857600,  # 100MB
                "lateral_movement_threshold": 5,
                "dns_tunnel_threshold": 1000,
                "beacon_threshold": 0.8,
                "traffic_anomaly_threshold": 3.0,
                "suspicious_ports": [22, 23, 135, 139, 445, 1433, 3389, 5900, 6667, 31337],
                "private_ip_ranges": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
                "enable_geolocation": True,
                "threat_intelligence": {
                    "enabled": False,
                    "api_key": "",
                    "sources": ["virustotal", "otx", "threatcrowd"]
                }
            },

            # Financial Forensics Settings
            "financial_fraud": {
                "amount_threshold": 10000.0,
                "velocity_threshold": 5,
                "unusual_time_hours": [0, 1, 2, 3, 4, 5],
                "structuring_thresholds": [9000, 9500, 9900],
                "round_amount_threshold": 5,
                "risk_score_weights": {
                    "amount": 0.3,
                    "velocity": 0.25,
                    "time": 0.15,
                    "location": 0.15,
                    "pattern": 0.15
                },
                "currencies": ["USD", "EUR", "GBP", "JPY", "CNY"],
                "regulatory_reporting": {
                    "enabled": False,
                    "threshold": 10000,
                    "jurisdiction": "US"
                }
            },

            # Behavioral Analysis Settings
            "behavioral": {
                "login_attempt_threshold": 5,
                "session_duration_threshold": 480,  # 8 hours in minutes
                "activity_burst_threshold": 5.0,
                "location_diversity_threshold": 0.5,
                "time_pattern_analysis": True,
                "user_baseline_days": 30,
                "anomaly_detection_methods": [
                    "statistical", "clustering", "isolation_forest", "svm"
                ]
            },

            # Machine Learning Settings
            "ml": {
                "ensemble_methods": ["random_forest", "xgboost", "isolation_forest"],
                "feature_selection": True,
                "auto_hyperparameter_tuning": False,
                "cross_validation_folds": 5,
                "test_size": 0.2,
                "random_state": 42,
                "n_jobs": -1,
                "model_persistence": True,
                "model_cache_directory": "models/cache",
                "training_data_sampling": {
                    "enabled": True,
                    "max_samples": 100000,
                    "stratified": True
                }
            },

            # Data Processing Settings
            "data": {
                "max_file_size_mb": 500,
                "supported_formats": [".csv", ".xlsx", ".json", ".parquet", ".log", ".txt"],
                "encoding_detection": True,
                "auto_data_types": True,
                "missing_value_threshold": 0.5,
                "outlier_detection_method": "iqr",
                "chunk_size": 10000,
                "memory_limit_gb": 8,
                "temp_directory": "temp/",
                "backup_original": True
            },

            # Visualization Settings
            "visualization": {
                "default_chart_type": "interactive",
                "color_palette": "viridis",
                "dark_theme_colors": {
                    "background": "#2d2d2d",
                    "text": "#ffffff",
                    "accent": "#0078d4",
                    "warning": "#ff8c00",
                    "danger": "#d13438",
                    "success": "#107c10"
                },
                "chart_export_dpi": 300,
                "interactive_charts": True,
                "enable_plotly": True,
                "enable_bokeh": True,
                "max_data_points": 10000,
                "animation_duration": 500
            },

            # Export Settings
            "export": {
                "default_format": "csv",
                "output_directory": "exports/",
                "include_metadata": True,
                "timestamp_in_filename": True,
                "compression": False,
                "pdf_settings": {
                    "page_size": "A4",
                    "orientation": "portrait",
                    "include_charts": True,
                    "chart_quality": "high"
                },
                "excel_settings": {
                    "include_formulas": False,
                    "freeze_panes": True,
                    "auto_filter": True,
                    "conditional_formatting": True
                }
            },

            # Logging Settings
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": True,
                "log_directory": "logs/",
                "max_log_size_mb": 10,
                "backup_count": 5,
                "console_logging": True,
                "log_sensitive_data": False
            },

            # Security Settings
            "security": {
                "enable_audit_trail": True,
                "mask_sensitive_data": True,
                "session_timeout_minutes": 60,
                "max_login_attempts": 3,
                "password_policy": {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_symbols": True
                },
                "data_encryption": {
                    "enabled": False,
                    "algorithm": "AES-256",
                    "key_rotation_days": 90
                }
            },

            # Performance Settings
            "performance": {
                "enable_multiprocessing": True,
                "max_workers": None,  # Auto-detect
                "memory_monitoring": True,
                "cpu_monitoring": True,
                "performance_profiling": False,
                "optimization_level": "balanced",  # conservative, balanced, aggressive
                "cache_size_mb": 512,
                "preload_models": True
            },

            # Integration Settings
            "integrations": {
                "database": {
                    "enabled": False,
                    "type": "sqlite",  # sqlite, postgresql, mysql
                    "connection_string": "",
                    "pool_size": 5
                },
                "apis": {
                    "enabled": False,
                    "rest_api": True,
                    "graphql_api": False,
                    "rate_limiting": True
                },
                "external_tools": {
                    "volatility": {"enabled": False, "path": ""},
                    "wireshark": {"enabled": False, "path": ""},
                    "sleuthkit": {"enabled": False, "path": ""}
                }
            }
        }

        # Load user configuration if it exists
        self.config_file = Path.home() / ".inspectra" / "config.json"
        self.load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'network.port_scan_threshold')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config

            for k in keys:
                value = value[k]

            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'network.port_scan_threshold')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def get_detection_thresholds(self, analysis_type: str) -> Dict[str, Any]:
        """
        Get detection thresholds for specific analysis type

        Args:
            analysis_type: Type of analysis ('network', 'financial_fraud', 'behavioral')

        Returns:
            Dictionary of thresholds for the analysis type
        """
        return self._config.get(analysis_type, {})

    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-specific settings"""
        return self._config.get("ui", {})

    def get_ml_settings(self) -> Dict[str, Any]:
        """Get machine learning settings"""
        return self._config.get("ml", {})

    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings"""
        return self._config.get("visualization", {})

    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # Merge user config with defaults
                self._merge_config(self._config, user_config)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.info("Using default configuration")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")

    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        # Backup current config
        backup_file = self.config_file.with_suffix('.backup.json')
        if self.config_file.exists():
            try:
                self.config_file.rename(backup_file)
                self.logger.info(f"Current config backed up to {backup_file}")
            except Exception as e:
                self.logger.warning(f"Could not backup config: {e}")

        # Reset to defaults
        self.__init__()
        self.logger.info("Configuration reset to defaults")

    def _merge_config(self, default_config: Dict, user_config: Dict) -> None:
        """
        Recursively merge user configuration with default configuration

        Args:
            default_config: Default configuration dictionary
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in default_config:
                if isinstance(default_config[key], dict) and isinstance(value, dict):
                    self._merge_config(default_config[key], value)
                else:
                    default_config[key] = value
            else:
                default_config[key] = value

    def validate_config(self) -> bool:
        """
        Validate configuration values

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate required sections
            required_sections = ['application', 'ui', 'analysis', 'network', 'financial_fraud']
            for section in required_sections:
                if section not in self._config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False

            # Validate data types and ranges
            validations = [
                ('ui.table_row_height', int, 20, 100),
                ('analysis.default_confidence_threshold', float, 0.0, 1.0),
                ('network.port_scan_threshold', int, 1, 1000),
                ('financial_fraud.amount_threshold', (int, float), 0, float('inf')),
                ('performance.cache_size_mb', int, 1, 10240),
            ]

            for key, expected_type, min_val, max_val in validations:
                value = self.get(key)
                if value is not None:
                    if not isinstance(value, expected_type):
                        self.logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                        return False

                    if isinstance(value, (int, float)):
                        if not (min_val <= value <= max_val):
                            self.logger.error(f"Value for {key} out of range: {value} (expected {min_val}-{max_val})")
                            return False

            self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    def export_config(self, export_path: Union[str, Path]) -> None:
        """
        Export current configuration to a file

        Args:
            export_path: Path to export configuration file
        """
        try:
            export_path = Path(export_path)

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration exported to {export_path}")

        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")

    def import_config(self, import_path: Union[str, Path]) -> None:
        """
        Import configuration from a file

        Args:
            import_path: Path to import configuration file
        """
        try:
            import_path = Path(import_path)

            if not import_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {import_path}")

            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            # Validate imported config
            temp_config = self._config.copy()
            self._merge_config(temp_config, imported_config)

            # If validation passes, apply the config
            self._config = temp_config
            self.logger.info(f"Configuration imported from {import_path}")

        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration

        Returns:
            Dictionary containing configuration summary
        """
        return {
            'application': self._config['application'],
            'sections': list(self._config.keys()),
            'total_settings': sum(len(section) if isinstance(section, dict) else 1
                                for section in self._config.values()),
            'config_file': str(self.config_file),
            'last_modified': self.config_file.stat().st_mtime if self.config_file.exists() else None
        }

# Global configuration instance
config = InspectraConfig()

# Convenience functions for common operations
def get_setting(key: str, default: Any = None) -> Any:
    """Get a configuration setting"""
    return config.get(key, default)

def set_setting(key: str, value: Any) -> None:
    """Set a configuration setting"""
    config.set(key, value)

def get_detection_thresholds(analysis_type: str) -> Dict[str, Any]:
    """Get detection thresholds for analysis type"""
    return config.get_detection_thresholds(analysis_type)

def save_config() -> None:
    """Save current configuration"""
    config.save_config()

def load_config() -> None:
    """Reload configuration from file"""
    config.load_config()