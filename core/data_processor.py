# data_processor.py
# Data preprocessing and cleaning functionality

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """Handles all data preprocessing and cleaning operations"""

    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.encoders = {}
        self.imputers = {}

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'total_cells': len(df) * len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'missing_data': {},
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'data_types': {},
            'column_stats': {}
        }

        # Missing data analysis
        missing_data = df.isnull().sum()
        report['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': {col: {
                'count': missing_data[col],
                'percentage': (missing_data[col] / len(df)) * 100
            } for col in missing_data[missing_data > 0].index}
        }

        # Data types analysis
        for dtype in df.dtypes.value_counts().index:
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            report['data_types'][str(dtype)] = {
                'count': len(cols),
                'columns': cols
            }

        # Column statistics
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                col_stats.update({
                    'mean': desc['mean'],
                    'std': desc['std'],
                    'min': desc['min'],
                    'max': desc['max'],
                    'median': desc['50%'],
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                })

                # Outlier detection using IQR
                Q1 = desc['25%']
                Q3 = desc['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                col_stats['outlier_count'] = len(outliers)
                col_stats['outlier_percentage'] = (len(outliers) / len(df)) * 100

            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                top_values = df[col].value_counts().head(5)
                col_stats.update({
                    'top_values': top_values.to_dict(),
                    'most_frequent': top_values.index[0] if len(top_values) > 0 else None,
                    'most_frequent_count': top_values.iloc[0] if len(top_values) > 0 else 0
                })

            report['column_stats'][col] = col_stats

        return report

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        df_clean = df.copy()

        # Remove leading/trailing whitespace
        df_clean.columns = df_clean.columns.str.strip()

        # Replace spaces and special characters with underscores
        df_clean.columns = df_clean.columns.str.replace(r'[^\w]', '_', regex=True)

        # Convert to lowercase
        df_clean.columns = df_clean.columns.str.lower()

        # Remove multiple consecutive underscores
        df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)

        # Remove leading/trailing underscores
        df_clean.columns = df_clean.columns.str.strip('_')

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto',
                              columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values with various strategies"""
        df_clean = df.copy()

        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            if df_clean[col].isnull().sum() == 0:
                continue

            if strategy == 'auto':
                # Auto-select strategy based on data type and missing percentage
                missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100

                if missing_pct > 50:
                    # Drop column if >50% missing
                    df_clean = df_clean.drop(columns=[col])
                    continue
                elif pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Use median for numeric
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Use mode for categorical
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna('Unknown')

            elif strategy == 'drop_rows':
                df_clean = df_clean.dropna(subset=[col])

            elif strategy == 'drop_columns':
                df_clean = df_clean.drop(columns=[col])

            elif strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

            elif strategy == 'mode':
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])

            elif strategy == 'forward_fill':
                df_clean[col] = df_clean[col].fillna(method='ffill')

            elif strategy == 'backward_fill':
                df_clean[col] = df_clean[col].fillna(method='bfill')

            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].interpolate()

        return df_clean

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                          keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep)

    def handle_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                        method: str = 'iqr', action: str = 'remove') -> pd.DataFrame:
        """Handle outliers using various methods"""
        df_clean = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                if action == 'remove':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                elif action == 'cap':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                elif action == 'replace_median':
                    median_val = df_clean[col].median()
                    df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = median_val

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                threshold = 3

                if action == 'remove':
                    outlier_mask = z_scores > threshold
                    df_clean = df_clean[~outlier_mask]
                elif action == 'cap':
                    # Cap at 3 standard deviations
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        return df_clean

    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                           method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            if method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))

            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

            elif method == 'ordinal':
                # Manual ordinal encoding (assumes ordered categories)
                unique_vals = sorted(df_encoded[col].unique())
                ordinal_map = {val: i for i, val in enumerate(unique_vals)}
                df_encoded[col] = df_encoded[col].map(ordinal_map)

        return df_encoded

    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method not in self.scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler = self.scalers[method]
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

        return df_scaled

    def create_date_features(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Extract features from date columns"""
        df_features = df.copy()

        for col in date_columns:
            if pd.api.types.is_datetime64_any_dtype(df_features[col]):
                # Extract date components
                df_features[f'{col}_year'] = df_features[col].dt.year
                df_features[f'{col}_month'] = df_features[col].dt.month
                df_features[f'{col}_day'] = df_features[col].dt.day
                df_features[f'{col}_dayofweek'] = df_features[col].dt.dayofweek
                df_features[f'{col}_quarter'] = df_features[col].dt.quarter
                df_features[f'{col}_is_weekend'] = df_features[col].dt.dayofweek.isin([5, 6]).astype(int)

                # Additional features
                df_features[f'{col}_days_since_epoch'] = (df_features[col] - pd.Timestamp('1970-01-01')).dt.days

        return df_features

    def detect_and_convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and convert appropriate data types"""
        df_converted = df.copy()

        for col in df_converted.columns:
            # Try to convert to numeric
            if df_converted[col].dtype == 'object':
                # Try datetime first
                try:
                    df_converted[col] = pd.to_datetime(df_converted[col])
                    continue
                except:
                    pass

                # Try numeric
                try:
                    # Check if it looks like numeric data
                    numeric_series = pd.to_numeric(df_converted[col], errors='coerce')
                    if numeric_series.notna().sum() / len(df_converted) > 0.8:  # 80% can be converted
                        df_converted[col] = numeric_series
                        continue
                except:
                    pass

                # Check for boolean-like strings
                unique_vals = df_converted[col].str.lower().unique()
                if len(unique_vals) <= 3 and all(val in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n', None]
                                                 for val in unique_vals):
                    bool_map = {'true': True, 'false': False, 'yes': True, 'no': False,
                                '1': True, '0': False, 'y': True, 'n': False}
                    df_converted[col] = df_converted[col].str.lower().map(bool_map)

        return df_converted

    def create_bins(self, df: pd.DataFrame, column: str, n_bins: int = 5,
                    method: str = 'equal_width') -> pd.DataFrame:
        """Create binned versions of continuous variables"""
        df_binned = df.copy()

        if method == 'equal_width':
            df_binned[f'{column}_binned'] = pd.cut(df_binned[column], bins=n_bins)
        elif method == 'equal_frequency':
            df_binned[f'{column}_binned'] = pd.qcut(df_binned[column], q=n_bins, duplicates='drop')

        return df_binned

    def feature_engineering(self, df: pd.DataFrame, operations: Dict[str, Any]) -> pd.DataFrame:
        """Perform various feature engineering operations"""
        df_engineered = df.copy()

        for operation, params in operations.items():
            if operation == 'polynomial':
                # Create polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                poly_columns = params.get('columns', [])
                degree = params.get('degree', 2)

                if poly_columns:
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_features = poly.fit_transform(df_engineered[poly_columns])
                    feature_names = poly.get_feature_names_out(poly_columns)

                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_engineered.index)
                    df_engineered = pd.concat([df_engineered, poly_df], axis=1)

            elif operation == 'interaction':
                # Create interaction features
                col1, col2 = params['columns']
                df_engineered[f'{col1}_x_{col2}'] = df_engineered[col1] * df_engineered[col2]

            elif operation == 'log_transform':
                # Log transformation
                columns = params.get('columns', [])
                for col in columns:
                    if (df_engineered[col] > 0).all():
                        df_engineered[f'{col}_log'] = np.log(df_engineered[col])

            elif operation == 'sqrt_transform':
                # Square root transformation
                columns = params.get('columns', [])
                for col in columns:
                    if (df_engineered[col] >= 0).all():
                        df_engineered[f'{col}_sqrt'] = np.sqrt(df_engineered[col])

        return df_engineered

    def correlation_filter(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        return df.drop(columns=to_drop)

    def apply_data_transformations(self, df: pd.DataFrame, transformations: Dict[str, Any]) -> pd.DataFrame:
        """Apply a pipeline of data transformations"""
        df_transformed = df.copy()

        # Apply transformations in order
        for transform, params in transformations.items():
            if transform == 'clean_names':
                df_transformed = self.clean_column_names(df_transformed)

            elif transform == 'handle_missing':
                strategy = params.get('strategy', 'auto')
                columns = params.get('columns', None)
                df_transformed = self.handle_missing_values(df_transformed, strategy, columns)

            elif transform == 'remove_duplicates':
                subset = params.get('subset', None)
                keep = params.get('keep', 'first')
                df_transformed = self.remove_duplicates(df_transformed, subset, keep)

            elif transform == 'handle_outliers':
                method = params.get('method', 'iqr')
                action = params.get('action', 'remove')
                columns = params.get('columns', None)
                df_transformed = self.handle_outliers(df_transformed, columns, method, action)

            elif transform == 'encode_categorical':
                method = params.get('method', 'label')
                columns = params.get('columns', None)
                df_transformed = self.encode_categorical(df_transformed, columns, method)

            elif transform == 'scale_features':
                method = params.get('method', 'standard')
                columns = params.get('columns', None)
                df_transformed = self.scale_features(df_transformed, columns, method)

            elif transform == 'convert_dtypes':
                df_transformed = self.detect_and_convert_dtypes(df_transformed)

        return df_transformed