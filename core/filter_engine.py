# filter_engine.py
# Advanced data filtering system for InSpectra Analytics

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import re
from datetime import datetime, date
import json
from pathlib import Path


class FilterEngine:
    """Advanced data filtering engine with multiple filter types and combinations"""

    def __init__(self):
        self.active_filters = {}
        self.filter_history = []
        self.saved_filters = {}
        self.original_data = None
        self.filtered_data = None

        # Define available filter operations
        self.filter_operations = {
            'equals': self._filter_equals,
            'not_equals': self._filter_not_equals,
            'contains': self._filter_contains,
            'not_contains': self._filter_not_contains,
            'starts_with': self._filter_starts_with,
            'ends_with': self._filter_ends_with,
            'greater_than': self._filter_greater_than,
            'less_than': self._filter_less_than,
            'greater_equal': self._filter_greater_equal,
            'less_equal': self._filter_less_equal,
            'between': self._filter_between,
            'not_between': self._filter_not_between,
            'in_list': self._filter_in_list,
            'not_in_list': self._filter_not_in_list,
            'is_null': self._filter_is_null,
            'is_not_null': self._filter_is_not_null,
            'regex': self._filter_regex,
            'date_equals': self._filter_date_equals,
            'date_after': self._filter_date_after,
            'date_before': self._filter_date_before,
            'date_between': self._filter_date_between,
            'top_n': self._filter_top_n,
            'bottom_n': self._filter_bottom_n,
            'outliers': self._filter_outliers,
            'duplicates': self._filter_duplicates,
            'unique': self._filter_unique
        }

    def set_data(self, df: pd.DataFrame):
        """Set the data to be filtered"""
        self.original_data = df.copy()
        self.filtered_data = df.copy()
        self.active_filters = {}

    def add_filter(self, filter_config: Dict[str, Any]) -> bool:
        """Add a new filter"""
        try:
            filter_id = filter_config.get('id', f"filter_{len(self.active_filters)}")

            # Validate filter configuration
            if not self._validate_filter_config(filter_config):
                return False

            # Store the filter
            self.active_filters[filter_id] = filter_config

            # Apply all filters
            self._apply_all_filters()

            # Add to history
            self.filter_history.append({
                'action': 'add',
                'filter_id': filter_id,
                'filter': filter_config.copy(),
                'timestamp': datetime.now(),
                'result_count': len(self.filtered_data)
            })

            return True

        except Exception as e:
            print(f"Error adding filter: {e}")
            return False

    def remove_filter(self, filter_id: str) -> bool:
        """Remove a specific filter"""
        try:
            if filter_id in self.active_filters:
                removed_filter = self.active_filters.pop(filter_id)

                # Reapply remaining filters
                self._apply_all_filters()

                # Add to history
                self.filter_history.append({
                    'action': 'remove',
                    'filter_id': filter_id,
                    'filter': removed_filter,
                    'timestamp': datetime.now(),
                    'result_count': len(self.filtered_data)
                })

                return True
            return False

        except Exception as e:
            print(f"Error removing filter: {e}")
            return False

    def clear_all_filters(self):
        """Clear all active filters"""
        try:
            self.active_filters = {}
            self.filtered_data = self.original_data.copy()

            # Add to history
            self.filter_history.append({
                'action': 'clear_all',
                'timestamp': datetime.now(),
                'result_count': len(self.filtered_data)
            })

        except Exception as e:
            print(f"Error clearing filters: {e}")

    def get_filtered_data(self) -> pd.DataFrame:
        """Get the current filtered data"""
        return self.filtered_data.copy() if self.filtered_data is not None else pd.DataFrame()

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of current filters and results"""
        if self.original_data is None:
            return {}

        original_count = len(self.original_data)
        filtered_count = len(self.filtered_data) if self.filtered_data is not None else 0

        return {
            'total_filters': len(self.active_filters),
            'original_rows': original_count,
            'filtered_rows': filtered_count,
            'filtered_percentage': (filtered_count / original_count * 100) if original_count > 0 else 0,
            'rows_removed': original_count - filtered_count,
            'active_filters': list(self.active_filters.keys())
        }

    def save_filter_set(self, name: str) -> bool:
        """Save current filter configuration"""
        try:
            self.saved_filters[name] = {
                'filters': self.active_filters.copy(),
                'created_at': datetime.now(),
                'description': f"Filter set with {len(self.active_filters)} filters"
            }
            return True
        except Exception as e:
            print(f"Error saving filter set: {e}")
            return False

    def load_filter_set(self, name: str) -> bool:
        """Load a saved filter configuration"""
        try:
            if name in self.saved_filters:
                self.active_filters = self.saved_filters[name]['filters'].copy()
                self._apply_all_filters()

                # Add to history
                self.filter_history.append({
                    'action': 'load_set',
                    'filter_set_name': name,
                    'timestamp': datetime.now(),
                    'result_count': len(self.filtered_data)
                })

                return True
            return False
        except Exception as e:
            print(f"Error loading filter set: {e}")
            return False

    def export_filter_config(self, filepath: str) -> bool:
        """Export filter configuration to file"""
        try:
            config = {
                'active_filters': self.active_filters,
                'saved_filters': self.saved_filters,
                'export_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Error exporting filter config: {e}")
            return False

    def import_filter_config(self, filepath: str) -> bool:
        """Import filter configuration from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.active_filters = config.get('active_filters', {})
            self.saved_filters = config.get('saved_filters', {})

            # Apply imported filters
            self._apply_all_filters()

            return True
        except Exception as e:
            print(f"Error importing filter config: {e}")
            return False

    def get_quick_filters(self, column: str) -> List[Dict[str, Any]]:
        """Get suggested quick filters for a column"""
        if self.original_data is None or column not in self.original_data.columns:
            return []

        col_data = self.original_data[column]
        quick_filters = []

        # Check data type and suggest appropriate filters
        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric column quick filters
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()

            quick_filters.extend([
                {
                    'name': f'Above Average ({mean_val:.2f})',
                    'config': {
                        'column': column,
                        'operation': 'greater_than',
                        'value': mean_val
                    }
                },
                {
                    'name': f'Below Average ({mean_val:.2f})',
                    'config': {
                        'column': column,
                        'operation': 'less_than',
                        'value': mean_val
                    }
                },
                {
                    'name': 'Top 10%',
                    'config': {
                        'column': column,
                        'operation': 'greater_equal',
                        'value': col_data.quantile(0.9)
                    }
                },
                {
                    'name': 'Bottom 10%',
                    'config': {
                        'column': column,
                        'operation': 'less_equal',
                        'value': col_data.quantile(0.1)
                    }
                }
            ])

            # Add outlier filters
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold_low = Q1 - 1.5 * IQR
            outlier_threshold_high = Q3 + 1.5 * IQR

            quick_filters.append({
                'name': 'Remove Outliers (IQR)',
                'config': {
                    'column': column,
                    'operation': 'between',
                    'value': [outlier_threshold_low, outlier_threshold_high]
                }
            })

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Date/time column quick filters
            min_date = col_data.min()
            max_date = col_data.max()

            quick_filters.extend([
                {
                    'name': 'Last 30 Days',
                    'config': {
                        'column': column,
                        'operation': 'date_after',
                        'value': (max_date - pd.Timedelta(days=30)).date()
                    }
                },
                {
                    'name': 'This Year',
                    'config': {
                        'column': column,
                        'operation': 'date_after',
                        'value': f"{max_date.year}-01-01"
                    }
                }
            ])

        else:
            # Categorical/text column quick filters
            value_counts = col_data.value_counts()
            top_values = value_counts.head(5)

            for value, count in top_values.items():
                if pd.notna(value):
                    quick_filters.append({
                        'name': f'Only "{value}" ({count} rows)',
                        'config': {
                            'column': column,
                            'operation': 'equals',
                            'value': value
                        }
                    })

        # Always add null/not null filters
        null_count = col_data.isnull().sum()
        if null_count > 0:
            quick_filters.append({
                'name': f'Remove Nulls ({null_count} rows)',
                'config': {
                    'column': column,
                    'operation': 'is_not_null'
                }
            })

        quick_filters.append({
            'name': f'Only Nulls ({null_count} rows)',
            'config': {
                'column': column,
                'operation': 'is_null'
            }
        })

        return quick_filters

    def _validate_filter_config(self, filter_config: Dict[str, Any]) -> bool:
        """Validate filter configuration"""
        required_fields = ['column', 'operation']

        for field in required_fields:
            if field not in filter_config:
                print(f"Missing required field: {field}")
                return False

        if filter_config['column'] not in self.original_data.columns:
            print(f"Column not found: {filter_config['column']}")
            return False

        if filter_config['operation'] not in self.filter_operations:
            print(f"Unknown operation: {filter_config['operation']}")
            return False

        return True

    def _apply_all_filters(self):
        """Apply all active filters to the data"""
        try:
            self.filtered_data = self.original_data.copy()

            for filter_id, filter_config in self.active_filters.items():
                column = filter_config['column']
                operation = filter_config['operation']

                # Get the filter function
                filter_func = self.filter_operations[operation]

                # Apply the filter
                mask = filter_func(self.filtered_data[column], filter_config)
                self.filtered_data = self.filtered_data[mask]

        except Exception as e:
            print(f"Error applying filters: {e}")
            self.filtered_data = self.original_data.copy()

    # Filter operation implementations
    def _filter_equals(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for exact equality"""
        value = config['value']
        case_sensitive = config.get('case_sensitive', True)

        if isinstance(value, str) and not case_sensitive:
            return series.astype(str).str.lower() == str(value).lower()
        return series == value

    def _filter_not_equals(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for inequality"""
        return ~self._filter_equals(series, config)

    def _filter_contains(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for string contains"""
        value = str(config['value'])
        case_sensitive = config.get('case_sensitive', True)

        if case_sensitive:
            return series.astype(str).str.contains(value, na=False)
        else:
            return series.astype(str).str.lower().str.contains(value.lower(), na=False)

    def _filter_not_contains(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for string does not contain"""
        return ~self._filter_contains(series, config)

    def _filter_starts_with(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for string starts with"""
        value = str(config['value'])
        case_sensitive = config.get('case_sensitive', True)

        if case_sensitive:
            return series.astype(str).str.startswith(value, na=False)
        else:
            return series.astype(str).str.lower().str.startswith(value.lower(), na=False)

    def _filter_ends_with(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for string ends with"""
        value = str(config['value'])
        case_sensitive = config.get('case_sensitive', True)

        if case_sensitive:
            return series.astype(str).str.endswith(value, na=False)
        else:
            return series.astype(str).str.lower().str.endswith(value.lower(), na=False)

    def _filter_greater_than(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for greater than"""
        return series > config['value']

    def _filter_less_than(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for less than"""
        return series < config['value']

    def _filter_greater_equal(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for greater than or equal"""
        return series >= config['value']

    def _filter_less_equal(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for less than or equal"""
        return series <= config['value']

    def _filter_between(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for between values (inclusive)"""
        value = config['value']
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (series >= value[0]) & (series <= value[1])
        return pd.Series([True] * len(series))

    def _filter_not_between(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for not between values"""
        return ~self._filter_between(series, config)

    def _filter_in_list(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for value in list"""
        value_list = config['value']
        if isinstance(value_list, (list, tuple)):
            return series.isin(value_list)
        return series == value_list

    def _filter_not_in_list(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for value not in list"""
        return ~self._filter_in_list(series, config)

    def _filter_is_null(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for null values"""
        return series.isnull()

    def _filter_is_not_null(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for non-null values"""
        return series.notnull()

    def _filter_regex(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter using regular expression"""
        pattern = config['value']
        flags = 0
        if not config.get('case_sensitive', True):
            flags |= re.IGNORECASE

        try:
            return series.astype(str).str.contains(pattern, regex=True, flags=flags, na=False)
        except Exception:
            return pd.Series([False] * len(series))

    def _filter_date_equals(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for date equals"""
        date_value = pd.to_datetime(config['value']).date()
        return pd.to_datetime(series).dt.date == date_value

    def _filter_date_after(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for date after"""
        date_value = pd.to_datetime(config['value'])
        return pd.to_datetime(series) > date_value

    def _filter_date_before(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for date before"""
        date_value = pd.to_datetime(config['value'])
        return pd.to_datetime(series) < date_value

    def _filter_date_between(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for date between"""
        value = config['value']
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start_date = pd.to_datetime(value[0])
            end_date = pd.to_datetime(value[1])
            dt_series = pd.to_datetime(series)
            return (dt_series >= start_date) & (dt_series <= end_date)
        return pd.Series([True] * len(series))

    def _filter_top_n(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for top N values"""
        n = config['value']
        if pd.api.types.is_numeric_dtype(series):
            threshold = series.nlargest(n).min()
            return series >= threshold
        else:
            top_values = series.value_counts().head(n).index
            return series.isin(top_values)

    def _filter_bottom_n(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for bottom N values"""
        n = config['value']
        if pd.api.types.is_numeric_dtype(series):
            threshold = series.nsmallest(n).max()
            return series <= threshold
        else:
            bottom_values = series.value_counts().tail(n).index
            return series.isin(bottom_values)

    def _filter_outliers(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([False] * len(series))

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (series < lower_bound) | (series > upper_bound)

    def _filter_duplicates(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for duplicate values"""
        return series.duplicated(keep=False)

    def _filter_unique(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Filter for unique values only"""
        return ~series.duplicated(keep=False)

    def get_filter_history(self) -> List[Dict[str, Any]]:
        """Get filter operation history"""
        return self.filter_history.copy()

    def get_available_operations(self, column: str) -> List[str]:
        """Get available filter operations for a column"""
        if self.original_data is None or column not in self.original_data.columns:
            return []

        col_data = self.original_data[column]
        operations = []

        # Basic operations available for all types
        operations.extend(['equals', 'not_equals', 'is_null', 'is_not_null', 'in_list', 'not_in_list'])

        # Numeric operations
        if pd.api.types.is_numeric_dtype(col_data):
            operations.extend([
                'greater_than', 'less_than', 'greater_equal', 'less_equal',
                'between', 'not_between', 'top_n', 'bottom_n', 'outliers'
            ])

        # String operations
        if col_data.dtype == 'object':
            operations.extend([
                'contains', 'not_contains', 'starts_with', 'ends_with', 'regex'
            ])

        # Date operations
        if pd.api.types.is_datetime64_any_dtype(col_data):
            operations.extend([
                'date_equals', 'date_after', 'date_before', 'date_between'
            ])

        # General operations
        operations.extend(['duplicates', 'unique'])

        return sorted(operations)

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get statistics for a column to help with filtering"""
        if self.original_data is None or column not in self.original_data.columns:
            return {}

        col_data = self.original_data[column]
        stats = {
            'column': column,
            'dtype': str(col_data.dtype),
            'total_count': len(col_data),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            })
        elif col_data.dtype == 'object':
            value_counts = col_data.value_counts().head(10)
            stats['top_values'] = value_counts.to_dict()
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            stats.update({
                'min_date': col_data.min(),
                'max_date': col_data.max(),
                'date_range_days': (col_data.max() - col_data.min()).days
            })

        return stats