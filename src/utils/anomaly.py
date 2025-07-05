"""
Anomaly detection utilities and data structures.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Anomaly:
    """
    Represents a detected anomaly in the data.
    
    Attributes:
        timestamp: When the anomaly occurred
        anomaly_type: Type/category of the anomaly
        severity: Severity level (low, medium, high, critical)
        description: Human-readable description
        confidence: Confidence score (0.0 to 1.0)
        source_data: Original data that triggered the anomaly
        metadata: Additional metadata about the anomaly
    """
    timestamp: datetime
    anomaly_type: str
    severity: str
    description: str
    confidence: float
    source_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate anomaly data after initialization."""
        # Ensure severity is valid
        valid_severities = ['low', 'medium', 'high', 'critical']
        if self.severity.lower() not in valid_severities:
            self.severity = 'medium'
        else:
            self.severity = self.severity.lower()
        
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'description': self.description,
            'confidence': self.confidence,
            'source_data': self.source_data,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Anomaly':
        """Create anomaly from dictionary data."""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        return cls(
            timestamp=timestamp,
            anomaly_type=data.get('anomaly_type', 'unknown'),
            severity=data.get('severity', 'medium'),
            description=data.get('description', 'No description available'),
            confidence=float(data.get('confidence', 0.5)),
            source_data=data.get('source_data', {}),
            metadata=data.get('metadata', {})
        )
    
    def get_severity_score(self) -> int:
        """Get numeric severity score for sorting."""
        severity_scores = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return severity_scores.get(self.severity, 2)
    
    def is_high_priority(self) -> bool:
        """Check if this anomaly is high priority."""
        return self.severity in ['high', 'critical'] and self.confidence >= 0.7
    
    def __str__(self) -> str:
        """String representation of the anomaly."""
        return f"[{self.severity.upper()}] {self.anomaly_type}: {self.description} (confidence: {self.confidence:.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Anomaly(type='{self.anomaly_type}', severity='{self.severity}', confidence={self.confidence:.2f})"


class AnomalyCollection:
    """
    Collection of anomalies with utility methods for analysis and filtering.
    """
    
    def __init__(self, anomalies: Optional[list] = None):
        """Initialize with optional list of anomalies."""
        self.anomalies = anomalies or []
    
    def add(self, anomaly: Anomaly):
        """Add an anomaly to the collection."""
        self.anomalies.append(anomaly)
    
    def extend(self, anomalies: list):
        """Add multiple anomalies to the collection."""
        self.anomalies.extend(anomalies)
    
    def filter_by_severity(self, severity: str) -> 'AnomalyCollection':
        """Filter anomalies by severity level."""
        filtered = [a for a in self.anomalies if a.severity == severity.lower()]
        return AnomalyCollection(filtered)
    
    def filter_by_type(self, anomaly_type: str) -> 'AnomalyCollection':
        """Filter anomalies by type."""
        filtered = [a for a in self.anomalies if a.anomaly_type == anomaly_type]
        return AnomalyCollection(filtered)
    
    def filter_by_confidence(self, min_confidence: float) -> 'AnomalyCollection':
        """Filter anomalies by minimum confidence score."""
        filtered = [a for a in self.anomalies if a.confidence >= min_confidence]
        return AnomalyCollection(filtered)
    
    def get_high_priority(self) -> 'AnomalyCollection':
        """Get high priority anomalies."""
        filtered = [a for a in self.anomalies if a.is_high_priority()]
        return AnomalyCollection(filtered)
    
    def sort_by_severity(self, reverse: bool = True) -> 'AnomalyCollection':
        """Sort anomalies by severity (highest first by default)."""
        sorted_anomalies = sorted(self.anomalies, 
                                key=lambda a: a.get_severity_score(), 
                                reverse=reverse)
        return AnomalyCollection(sorted_anomalies)
    
    def sort_by_confidence(self, reverse: bool = True) -> 'AnomalyCollection':
        """Sort anomalies by confidence (highest first by default)."""
        sorted_anomalies = sorted(self.anomalies, 
                                key=lambda a: a.confidence, 
                                reverse=reverse)
        return AnomalyCollection(sorted_anomalies)
    
    def sort_by_timestamp(self, reverse: bool = True) -> 'AnomalyCollection':
        """Sort anomalies by timestamp (newest first by default)."""
        sorted_anomalies = sorted(self.anomalies, 
                                key=lambda a: a.timestamp, 
                                reverse=reverse)
        return AnomalyCollection(sorted_anomalies)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the anomaly collection."""
        if not self.anomalies:
            return {
                'total': 0,
                'by_severity': {},
                'by_type': {},
                'avg_confidence': 0.0,
                'high_priority_count': 0
            }
        
        # Count by severity
        severity_counts = {}
        for anomaly in self.anomalies:
            severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for anomaly in self.anomalies:
            type_counts[anomaly.anomaly_type] = type_counts.get(anomaly.anomaly_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(a.confidence for a in self.anomalies) / len(self.anomalies)
        
        # Count high priority
        high_priority_count = len([a for a in self.anomalies if a.is_high_priority()])
        
        return {
            'total': len(self.anomalies),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'avg_confidence': avg_confidence,
            'high_priority_count': high_priority_count
        }
    
    def to_list(self) -> list:
        """Convert to list of anomalies."""
        return self.anomalies.copy()
    
    def to_dict_list(self) -> list:
        """Convert to list of dictionaries."""
        return [anomaly.to_dict() for anomaly in self.anomalies]
    
    def __len__(self) -> int:
        """Get number of anomalies in collection."""
        return len(self.anomalies)
    
    def __iter__(self):
        """Make collection iterable."""
        return iter(self.anomalies)
    
    def __getitem__(self, index):
        """Allow indexing into the collection."""
        return self.anomalies[index]
    
    def __bool__(self) -> bool:
        """Check if collection has any anomalies."""
        return len(self.anomalies) > 0


def create_anomaly(timestamp, anomaly_type: str, severity: str, description: str, 
                  confidence: float, source_data: Dict[str, Any], 
                  metadata: Optional[Dict[str, Any]] = None) -> Anomaly:
    """
    Convenience function to create an anomaly.
    
    Args:
        timestamp: When the anomaly occurred
        anomaly_type: Type/category of the anomaly
        severity: Severity level (low, medium, high, critical)
        description: Human-readable description
        confidence: Confidence score (0.0 to 1.0)
        source_data: Original data that triggered the anomaly
        metadata: Additional metadata about the anomaly
    
    Returns:
        Anomaly object
    """
    if not isinstance(timestamp, datetime):
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
    
    return Anomaly(
        timestamp=timestamp,
        anomaly_type=anomaly_type,
        severity=severity,
        description=description,
        confidence=confidence,
        source_data=source_data,
        metadata=metadata
    )


def create_network_anomaly(timestamp, detection_type: str, severity: str, 
                          description: str, confidence: float, 
                          source_ip: str, dest_ip: str, port: int,
                          additional_data: Optional[Dict[str, Any]] = None) -> Anomaly:
    """
    Create a network-specific anomaly.
    
    Args:
        timestamp: When the anomaly occurred
        detection_type: Type of network anomaly
        severity: Severity level
        description: Description of the anomaly
        confidence: Confidence score
        source_ip: Source IP address
        dest_ip: Destination IP address
        port: Port number
        additional_data: Additional network data
    
    Returns:
        Anomaly object with network-specific metadata
    """
    source_data = {
        'source_ip': source_ip,
        'destination_ip': dest_ip,
        'port': port
    }
    
    if additional_data:
        source_data.update(additional_data)
    
    metadata = {
        'category': 'network',
        'detection_method': detection_type
    }
    
    return create_anomaly(
        timestamp=timestamp,
        anomaly_type=f"network_{detection_type}",
        severity=severity,
        description=description,
        confidence=confidence,
        source_data=source_data,
        metadata=metadata
    )


def create_financial_anomaly(timestamp, detection_type: str, severity: str,
                           description: str, confidence: float,
                           user_id: str, amount: float, transaction_type: str,
                           additional_data: Optional[Dict[str, Any]] = None) -> Anomaly:
    """
    Create a financial-specific anomaly.
    
    Args:
        timestamp: When the anomaly occurred
        detection_type: Type of financial anomaly
        severity: Severity level
        description: Description of the anomaly
        confidence: Confidence score
        user_id: User identifier
        amount: Transaction amount
        transaction_type: Type of transaction
        additional_data: Additional financial data
    
    Returns:
        Anomaly object with financial-specific metadata
    """
    source_data = {
        'user_id': user_id,
        'amount': amount,
        'transaction_type': transaction_type
    }
    
    if additional_data:
        source_data.update(additional_data)
    
    metadata = {
        'category': 'financial',
        'detection_method': detection_type
    }
    
    return create_anomaly(
        timestamp=timestamp,
        anomaly_type=f"financial_{detection_type}",
        severity=severity,
        description=description,
        confidence=confidence,
        source_data=source_data,
        metadata=metadata
    )

