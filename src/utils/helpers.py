"""
Utility helper functions for CyberForensics Data Detective.
"""

import hashlib
import ipaddress
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup application logging."""
    logger = logging.getLogger("CyberForensics")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def validate_ip_address(ip: str) -> bool:
    """Validate if string is a valid IP address."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_private_ip(ip: str) -> bool:
    """Check if IP address is private."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError:
        return False


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def parse_timestamp(timestamp: Union[str, int, float]) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError):
            return None
    
    if isinstance(timestamp, str):
        # Common timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y%m%d %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
    
    return None


def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL."""
    pattern = r'https?://(?:www\.)?([^/]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def is_suspicious_port(port: int) -> bool:
    """Check if port is commonly associated with suspicious activity."""
    suspicious_ports = {
        22: "SSH",
        23: "Telnet", 
        135: "RPC",
        139: "NetBIOS",
        445: "SMB",
        1433: "SQL Server",
        3389: "RDP",
        5900: "VNC",
        6667: "IRC",
        31337: "Back Orifice"
    }
    return port in suspicious_ports


def detect_anomalous_amounts(amounts: List[float], threshold_multiplier: float = 3.0) -> List[int]:
    """Detect anomalous amounts using statistical methods."""
    if len(amounts) < 3:
        return []
    
    df = pd.Series(amounts)
    mean = df.mean()
    std = df.std()
    
    if std == 0:
        return []
    
    threshold = mean + (threshold_multiplier * std)
    anomalies = df[df > threshold].index.tolist()
    
    return anomalies


def calculate_velocity(timestamps: List[datetime], window_minutes: int = 60) -> Dict[datetime, int]:
    """Calculate transaction velocity (count per time window)."""
    if not timestamps:
        return {}
    
    timestamps = sorted(timestamps)
    velocity = {}
    window_delta = timedelta(minutes=window_minutes)
    
    for i, ts in enumerate(timestamps):
        window_start = ts - window_delta
        count = sum(1 for t in timestamps[max(0, i-100):i+1] if t >= window_start)
        velocity[ts] = count
    
    return velocity


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


def create_report_filename(report_type: str, timestamp: Optional[datetime] = None) -> str:
    """Create standardized report filename."""
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{report_type}_report_{timestamp_str}.csv"


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """Split DataFrame into chunks for processing."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])
    return chunks


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value for division by zero."""
    try:
        return a / b if b != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def extract_features_from_text(text: str) -> Dict[str, Any]:
    """Extract basic features from text for analysis."""
    if not text:
        return {}
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'digit_count': sum(c.isdigit() for c in text),
        'upper_count': sum(c.isupper() for c in text),
        'special_char_count': sum(not c.isalnum() and not c.isspace() for c in text),
        'has_email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'has_url': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'has_ip': bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text))
    }




def format_timestamp(timestamp) -> str:
    """Format timestamp for display."""
    if pd.isna(timestamp):
        return "N/A"
    
    if isinstance(timestamp, str):
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            return str(timestamp)
    
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    import math
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

