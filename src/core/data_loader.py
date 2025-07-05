"""
Data loader module for CyberForensics Data Detective.
Handles loading various data formats commonly used in cybersecurity and forensics.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime

from utils.helpers import parse_timestamp, validate_ip_address


class DataLoader:
    """Universal data loader for cybersecurity and forensics data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.log': self._load_log,
            '.txt': self._load_text,
            '.xlsx': self._load_excel,
            '.parquet': self._load_parquet
        }

    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from file based on extension."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")

        self.logger.info(f"Loading data from {file_path}")

        try:
            df = self.supported_formats[extension](file_path, **kwargs)
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise

    def load_data_with_info(self, file_path: Union[str, Path], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data from file and return both data and file information."""
        file_path = Path(file_path)

        # Load the data
        df = self.load_data(file_path, **kwargs)

        # Create file info dictionary
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.to_dict()
        }

        return df, file_info

    def load_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Alias for load_data method for compatibility."""
        return self.load_data(file_path, **kwargs)

    def load_file_with_info(self, file_path: Union[str, Path], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Alias for load_data_with_info method for compatibility."""
        return self.load_data_with_info(file_path, **kwargs)

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                return self._standardize_dataframe(df)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode {file_path} with any supported encoding")

    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find the main data array
            for key in ['data', 'records', 'events', 'logs']:
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                # Flatten the dictionary
                df = pd.json_normalize(data)
        else:
            raise ValueError("Unsupported JSON structure")

        return self._standardize_dataframe(df)

    def _load_log(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load log file with intelligent parsing."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Detect log format
        log_format = self._detect_log_format(lines[:10])

        if log_format == 'apache':
            return self._parse_apache_logs(lines)
        elif log_format == 'iis':
            return self._parse_iis_logs(lines)
        elif log_format == 'syslog':
            return self._parse_syslog(lines)
        elif log_format == 'windows_event':
            return self._parse_windows_event_logs(lines)
        else:
            return self._parse_generic_logs(lines)

    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load text file as log file."""
        return self._load_log(file_path, **kwargs)

    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        try:
            df = pd.read_excel(file_path, **kwargs)
            return self._standardize_dataframe(df)
        except ImportError:
            raise ImportError("openpyxl is required to read Excel files")

    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(file_path, **kwargs)
            return self._standardize_dataframe(df)
        except ImportError:
            raise ImportError("pyarrow is required to read Parquet files")

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame columns and data types."""
        # Convert column names to lowercase and replace spaces with underscores
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]

        # Try to parse timestamp columns
        timestamp_columns = ['timestamp', 'time', 'datetime', 'date', 'created_at', 'updated_at']
        for col in df.columns:
            if any(ts_col in col for ts_col in timestamp_columns):
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError, pd.errors.ParserError):
                    # If conversion fails, keep original data
                    pass

        # Try to identify IP address columns
        ip_columns = ['ip', 'source_ip', 'dest_ip', 'client_ip', 'server_ip', 'remote_addr']
        for col in df.columns:
            if any(ip_col in col for ip_col in ip_columns):
                # Validate IP addresses
                if df[col].dtype == 'object':
                    df[f'{col}_valid'] = df[col].apply(lambda x: validate_ip_address(str(x)) if pd.notna(x) else False)

        return df

    def _detect_log_format(self, sample_lines: List[str]) -> str:
        """Detect log format from sample lines."""
        sample_text = '\n'.join(sample_lines)

        # Apache/Nginx access log
        if re.search(r'\d+\.\d+\.\d+\.\d+ - - \[', sample_text):
            return 'apache'

        # IIS log
        if re.search(r'#Software: Microsoft Internet Information Services', sample_text):
            return 'iis'

        # Syslog
        if re.search(r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', sample_text):
            return 'syslog'

        # Windows Event Log
        if re.search(r'EventID|Event ID', sample_text):
            return 'windows_event'

        return 'generic'

    def _parse_apache_logs(self, lines: List[str]) -> pd.DataFrame:
        """Parse Apache/Nginx access logs."""
        pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)"'

        records = []
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                records.append({
                    'ip_address': match.group(1),
                    'timestamp': match.group(2),
                    'request': match.group(3),
                    'status_code': int(match.group(4)),
                    'response_size': int(match.group(5)) if match.group(5) != '-' else 0,
                    'referer': match.group(6),
                    'user_agent': match.group(7)
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

        return df

    def _parse_iis_logs(self, lines: List[str]) -> pd.DataFrame:
        """Parse IIS logs."""
        # Find header line
        header_line = None
        data_lines = []

        for line in lines:
            if line.startswith('#Fields:'):
                header_line = line[8:].strip().split()
            elif not line.startswith('#') and line.strip():
                data_lines.append(line.strip().split())

        if header_line and data_lines:
            df = pd.DataFrame(data_lines, columns=header_line)
            # Convert timestamp columns
            if 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            return df

        return pd.DataFrame()

    def _parse_syslog(self, lines: List[str]) -> pd.DataFrame:
        """Parse syslog format."""
        pattern = r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s*(.*)'

        records = []
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                records.append({
                    'timestamp': match.group(1),
                    'hostname': match.group(2),
                    'process': match.group(3),
                    'message': match.group(4)
                })

        df = pd.DataFrame(records)
        if not df.empty:
            # Add current year to timestamp
            current_year = datetime.now().year
            df['timestamp'] = pd.to_datetime(f"{current_year} " + df['timestamp'], format='%Y %b %d %H:%M:%S', errors='coerce')

        return df

    def _parse_windows_event_logs(self, lines: List[str]) -> pd.DataFrame:
        """Parse Windows Event logs (simplified)."""
        records = []
        current_record = {}

        for line in lines:
            line = line.strip()
            if 'Event ID' in line or 'EventID' in line:
                if current_record:
                    records.append(current_record)
                current_record = {'event_id': line}
            elif 'Source:' in line:
                current_record['source'] = line.replace('Source:', '').strip()
            elif 'Time:' in line or 'Date:' in line:
                current_record['timestamp'] = line.replace('Time:', '').replace('Date:', '').strip()
            elif line:
                current_record['message'] = current_record.get('message', '') + ' ' + line

        if current_record:
            records.append(current_record)

        return pd.DataFrame(records)

    def _parse_generic_logs(self, lines: List[str]) -> pd.DataFrame:
        """Parse generic log format."""
        records = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                # Try to extract timestamp from beginning of line
                timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})', line)

                record = {
                    'line_number': i + 1,
                    'raw_line': line
                }

                if timestamp_match:
                    record['timestamp'] = timestamp_match.group(1)
                    record['message'] = line[len(timestamp_match.group(1)):].strip()
                else:
                    record['message'] = line

                records.append(record)

        df = pd.DataFrame(records)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        return df

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys())

    def preview_data(self, file_path: Union[str, Path], num_rows: int = 5) -> pd.DataFrame:
        """Preview first few rows of data."""
        df = self.load_data(file_path)
        return df.head(num_rows)