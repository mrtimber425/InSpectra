"""
Data processor module for CyberForensics Data Detective.
Handles data cleaning, preprocessing, and feature engineering for cybersecurity analysis.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.helpers import validate_ip_address, is_private_ip, extract_features_from_text


class DataProcessor:
    """Data processor for cybersecurity and forensics data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_data(self, data: pd.DataFrame, analysis_type: str = 'general') -> pd.DataFrame:
        """Process data for analysis."""
        self.logger.info(f"Processing data for {analysis_type} analysis")
        
        processed_data = data.copy()
        
        # Apply processing based on analysis type
        if analysis_type == 'network':
            processed_data = self.extract_network_features(processed_data)
        elif analysis_type == 'financial':
            processed_data = self.extract_financial_features(processed_data)
        elif analysis_type == 'behavioral':
            processed_data = self.extract_behavioral_features(processed_data)
        
        # Apply general cleaning
        processed_data = self.clean_data(processed_data)
        
        return processed_data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Standardize column names
        df_clean.columns = [self._standardize_column_name(col) for col in df_clean.columns]
        
        # Clean IP addresses
        df_clean = self._clean_ip_addresses(df_clean)
        
        # Clean timestamps
        df_clean = self._clean_timestamps(df_clean)
        
        # Clean numeric columns
        df_clean = self._clean_numeric_columns(df_clean)
        
        # Clean text columns
        df_clean = self._clean_text_columns(df_clean)
        
        self.logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract network-specific features for analysis."""
        self.logger.info("Extracting network features")
        
        df_features = df.copy()
        
        # IP address features
        ip_columns = [col for col in df.columns if 'ip' in col.lower()]
        for col in ip_columns:
            if df[col].dtype == 'object':
                df_features[f'{col}_is_private'] = df[col].apply(
                    lambda x: is_private_ip(str(x)) if pd.notna(x) else False
                )
                df_features[f'{col}_is_valid'] = df[col].apply(
                    lambda x: validate_ip_address(str(x)) if pd.notna(x) else False
                )
        
        # Port features
        port_columns = [col for col in df.columns if 'port' in col.lower()]
        for col in port_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_features[f'{col}_is_well_known'] = df[col].apply(
                    lambda x: x <= 1023 if pd.notna(x) else False
                )
                df_features[f'{col}_is_suspicious'] = df[col].apply(
                    lambda x: self._is_suspicious_port(x) if pd.notna(x) else False
                )
        
        # Protocol features
        protocol_columns = [col for col in df.columns if 'protocol' in col.lower()]
        for col in protocol_columns:
            if df[col].dtype == 'object':
                df_features[f'{col}_normalized'] = df[col].str.upper()
        
        # Packet size features
        size_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['size', 'length', 'bytes'])]
        for col in size_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_features[f'{col}_category'] = pd.cut(
                    df[col], 
                    bins=[0, 64, 512, 1500, 65535], 
                    labels=['tiny', 'small', 'medium', 'large']
                )
        
        return df_features
    
    def extract_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract financial transaction features for fraud detection."""
        self.logger.info("Extracting financial features")
        
        df_features = df.copy()
        
        # Amount features
        amount_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['amount', 'value', 'sum', 'total'])]
        for col in amount_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Amount categories
                df_features[f'{col}_category'] = pd.cut(
                    df[col],
                    bins=[0, 100, 1000, 10000, float('inf')],
                    labels=['small', 'medium', 'large', 'very_large']
                )
                
                # Round amounts (potential indicator of fraud)
                df_features[f'{col}_is_round'] = (df[col] % 100 == 0) & (df[col] > 0)
        
        # Time-based features
        timestamp_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'created'])]
        for col in timestamp_columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df_features[f'{col}_hour'] = df[col].dt.hour
                df_features[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df_features[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
                df_features[f'{col}_is_night'] = (df[col].dt.hour >= 22) | (df[col].dt.hour <= 6)
        
        # Account/User features
        account_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['account', 'user', 'customer'])]
        for col in account_columns:
            if df[col].dtype == 'object':
                # Transaction frequency per account
                df_features[f'{col}_transaction_count'] = df.groupby(col)[col].transform('count')
        
        return df_features
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features for anomaly detection."""
        self.logger.info("Extracting behavioral features")
        
        df_features = df.copy()
        
        # Session features
        session_columns = [col for col in df.columns if 'session' in col.lower()]
        for col in session_columns:
            if df[col].dtype == 'object':
                df_features[f'{col}_length'] = df[col].str.len()
        
        # User agent features
        ua_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['user_agent', 'useragent', 'agent'])]
        for col in ua_columns:
            if df[col].dtype == 'object':
                df_features[f'{col}_features'] = df[col].apply(self._extract_user_agent_features)
        
        # URL features
        url_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['url', 'uri', 'path'])]
        for col in url_columns:
            if df[col].dtype == 'object':
                df_features[f'{col}_length'] = df[col].str.len()
                df_features[f'{col}_has_params'] = df[col].str.contains(r'\?', na=False)
                df_features[f'{col}_suspicious_chars'] = df[col].apply(self._count_suspicious_url_chars)
        
        return df_features
    
    def calculate_time_windows(self, df: pd.DataFrame, timestamp_col: str, window_minutes: int = 60) -> pd.DataFrame:
        """Calculate time-based aggregations."""
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df
        
        df_windowed = df.copy()
        
        # Sort by timestamp
        df_windowed = df_windowed.sort_values(timestamp_col)
        
        # Create time windows
        df_windowed['time_window'] = df_windowed[timestamp_col].dt.floor(f'{window_minutes}min')
        
        # Calculate window statistics
        window_stats = df_windowed.groupby('time_window').agg({
            timestamp_col: 'count'
        }).rename(columns={timestamp_col: 'events_in_window'})
        
        df_windowed = df_windowed.merge(window_stats, on='time_window', how='left')
        
        return df_windowed
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers in specified columns."""
        df_outliers = df.copy()
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_outliers[f'{col}_is_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_outliers[f'{col}_is_outlier'] = z_scores > 3
        
        return df_outliers
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Standardize column name."""
        # Convert to lowercase and replace spaces/special chars with underscores
        standardized = re.sub(r'[^a-zA-Z0-9_]', '_', col_name.lower())
        # Remove multiple consecutive underscores
        standardized = re.sub(r'_+', '_', standardized)
        # Remove leading/trailing underscores
        return standardized.strip('_')
    
    def _clean_ip_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean IP address columns."""
        ip_columns = [col for col in df.columns if 'ip' in col.lower()]
        
        for col in ip_columns:
            if df[col].dtype == 'object':
                # Remove common prefixes/suffixes
                df[col] = df[col].str.replace(r'^.*?(\d+\.\d+\.\d+\.\d+).*$', r'\1', regex=True)
                # Validate and clean
                df[col] = df[col].apply(lambda x: x if validate_ip_address(str(x)) else np.nan)
        
        return df
    
    def _clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean timestamp columns."""
        timestamp_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'updated'])]
        
        for col in timestamp_columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks like numbers
                if df[col].str.match(r'^-?\d+\.?\d*$', na=False).any():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns."""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Remove leading/trailing whitespace
            df[col] = df[col].str.strip()
            # Replace empty strings with NaN
            df[col] = df[col].replace('', np.nan)
        
        return df
    
    def _is_suspicious_port(self, port: int) -> bool:
        """Check if port is suspicious."""
        suspicious_ports = {22, 23, 135, 139, 445, 1433, 3389, 5900, 6667, 31337}
        return port in suspicious_ports
    
    def _extract_user_agent_features(self, user_agent: str) -> Dict[str, Any]:
        """Extract features from user agent string."""
        if pd.isna(user_agent):
            return {}
        
        ua = str(user_agent)
        return {
            'length': len(ua),
            'is_bot': any(bot in ua.lower() for bot in ['bot', 'crawler', 'spider', 'scraper']),
            'has_version': bool(re.search(r'\d+\.\d+', ua)),
            'browser_count': len(re.findall(r'(Chrome|Firefox|Safari|Edge|Opera)', ua)),
            'suspicious_chars': sum(c in ua for c in '<>{}[]|\\')
        }
    
    def _count_suspicious_url_chars(self, url: str) -> int:
        """Count suspicious characters in URL."""
        if pd.isna(url):
            return 0
        
        suspicious_chars = ['<', '>', '{', '}', '|', '\\', '^', '`', '"']
        return sum(char in str(url) for char in suspicious_chars)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'text_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime']).columns)
        }
        
        return summary

