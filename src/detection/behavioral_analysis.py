"""
Behavioral analysis module for CyberForensics Data Detective.
Analyzes user behavior patterns to detect anomalies and suspicious activities.
"""

import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.helpers import extract_features_from_text
from config import config


@dataclass
class BehavioralAnomaly:
    """Represents a detected behavioral anomaly."""
    timestamp: datetime
    anomaly_type: str
    severity: str
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    confidence: float
    raw_data: Dict[str, Any]


class BehavioralAnalyzer:
    """Analyzer for user behavioral patterns and anomalies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.thresholds = config.get_detection_thresholds('behavioral')
        
        self.detection_methods = {
            'login_anomalies': self._detect_login_anomalies,
            'session_anomalies': self._detect_session_anomalies,
            'access_pattern_anomalies': self._detect_access_pattern_anomalies,
            'privilege_escalation': self._detect_privilege_escalation,
            'data_access_anomalies': self._detect_data_access_anomalies
        }
    
    def analyze_behavioral_data(self, df: pd.DataFrame, detection_types: Optional[List[str]] = None) -> List[BehavioralAnomaly]:
        """Analyze behavioral data for anomalies."""
        self.logger.info(f"Starting behavioral analysis on {len(df)} records")
        
        if detection_types is None:
            detection_types = list(self.detection_methods.keys())
        
        all_anomalies = []
        
        for detection_type in detection_types:
            if detection_type in self.detection_methods:
                self.logger.info(f"Running {detection_type} detection")
                try:
                    anomalies = self.detection_methods[detection_type](df)
                    all_anomalies.extend(anomalies)
                    self.logger.info(f"Found {len(anomalies)} {detection_type} anomalies")
                except Exception as e:
                    self.logger.error(f"Error in {detection_type} detection: {e}")
        
        # Sort by severity and timestamp
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_anomalies.sort(key=lambda x: (severity_order.get(x.severity, 4), x.timestamp))
        
        self.logger.info(f"Total behavioral anomalies detected: {len(all_anomalies)}")
        return all_anomalies
    
    def _detect_login_anomalies(self, df: pd.DataFrame) -> List[BehavioralAnomaly]:
        """Detect login-related anomalies."""
        anomalies = []
        
        # Look for failed login attempts
        if 'status' in df.columns or 'result' in df.columns:
            status_col = 'status' if 'status' in df.columns else 'result'
            failed_logins = df[df[status_col].str.contains('fail|error|denied', case=False, na=False)]
            
            if not failed_logins.empty and 'user_id' in df.columns:
                # Count failed attempts per user
                failed_counts = failed_logins.groupby('user_id').size()
                threshold = self.thresholds.get('login_attempt_threshold', 5)
                
                suspicious_users = failed_counts[failed_counts > threshold]
                
                for user_id, count in suspicious_users.items():
                    user_failures = failed_logins[failed_logins['user_id'] == user_id]
                    first_attempt = user_failures['timestamp'].min() if 'timestamp' in df.columns else datetime.now()
                    
                    severity = 'critical' if count > threshold * 3 else 'high'
                    confidence = min(0.9, count / (threshold * 2))
                    
                    anomaly = BehavioralAnomaly(
                        timestamp=first_attempt,
                        anomaly_type='login_anomaly',
                        severity=severity,
                        user_id=str(user_id),
                        session_id=None,
                        description=f"Multiple failed login attempts: {count} failures",
                        confidence=confidence,
                        raw_data={
                            'failed_attempts': int(count),
                            'threshold': threshold,
                            'time_span_hours': (user_failures['timestamp'].max() - user_failures['timestamp'].min()).total_seconds() / 3600 if 'timestamp' in df.columns else 0
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_session_anomalies(self, df: pd.DataFrame) -> List[BehavioralAnomaly]:
        """Detect session-related anomalies."""
        anomalies = []
        
        if 'session_id' not in df.columns:
            return anomalies
        
        # Analyze session durations
        if 'timestamp' in df.columns:
            session_stats = df.groupby('session_id')['timestamp'].agg(['min', 'max', 'count']).reset_index()
            session_stats['duration_minutes'] = (session_stats['max'] - session_stats['min']).dt.total_seconds() / 60
            
            # Detect unusually long sessions
            duration_threshold = self.thresholds.get('session_duration_threshold', 480)  # 8 hours
            long_sessions = session_stats[session_stats['duration_minutes'] > duration_threshold]
            
            for _, session in long_sessions.iterrows():
                session_id = session['session_id']
                duration = session['duration_minutes']
                
                # Get user info if available
                session_data = df[df['session_id'] == session_id]
                user_id = session_data['user_id'].iloc[0] if 'user_id' in df.columns and not session_data.empty else 'Unknown'
                
                severity = 'high' if duration > duration_threshold * 2 else 'medium'
                confidence = min(0.8, duration / (duration_threshold * 3))
                
                anomaly = BehavioralAnomaly(
                    timestamp=session['min'],
                    anomaly_type='session_anomaly',
                    severity=severity,
                    user_id=str(user_id),
                    session_id=str(session_id),
                    description=f"Unusually long session: {duration:.1f} minutes",
                    confidence=confidence,
                    raw_data={
                        'duration_minutes': float(duration),
                        'threshold_minutes': duration_threshold,
                        'activity_count': int(session['count'])
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_access_pattern_anomalies(self, df: pd.DataFrame) -> List[BehavioralAnomaly]:
        """Detect unusual access patterns."""
        anomalies = []
        
        if 'user_id' not in df.columns:
            return anomalies
        
        # Analyze access times
        if 'timestamp' in df.columns:
            df_time = df.copy()
            df_time['hour'] = df_time['timestamp'].dt.hour
            df_time['day_of_week'] = df_time['timestamp'].dt.dayofweek
            
            # For each user, find their typical access patterns
            for user_id, user_data in df_time.groupby('user_id'):
                if len(user_data) < 10:  # Need sufficient data
                    continue
                
                # Analyze typical hours
                typical_hours = user_data['hour'].value_counts()
                total_accesses = len(user_data)
                
                # Find accesses outside typical hours (less than 5% of total)
                unusual_accesses = user_data[
                    user_data['hour'].apply(lambda h: typical_hours.get(h, 0) / total_accesses < 0.05)
                ]
                
                for _, access in unusual_accesses.iterrows():
                    timestamp = access['timestamp']
                    hour = access['hour']
                    
                    severity = 'medium' if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 'low'
                    confidence = 0.6
                    
                    anomaly = BehavioralAnomaly(
                        timestamp=timestamp,
                        anomaly_type='access_pattern_anomaly',
                        severity=severity,
                        user_id=str(user_id),
                        session_id=access.get('session_id'),
                        description=f"Access at unusual time: {hour:02d}:xx (typical pattern deviation)",
                        confidence=confidence,
                        raw_data={
                            'hour': int(hour),
                            'typical_hour_frequency': float(typical_hours.get(hour, 0) / total_accesses),
                            'day_of_week': int(access['day_of_week'])
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_privilege_escalation(self, df: pd.DataFrame) -> List[BehavioralAnomaly]:
        """Detect privilege escalation attempts."""
        anomalies = []
        
        # Look for privilege-related columns
        privilege_columns = ['privilege', 'role', 'permission', 'access_level']
        privilege_col = None
        
        for col in privilege_columns:
            if col in df.columns:
                privilege_col = col
                break
        
        if not privilege_col or 'user_id' not in df.columns:
            return anomalies
        
        # Track privilege changes for each user
        for user_id, user_data in df.groupby('user_id'):
            if len(user_data) < 2:
                continue
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                user_data_sorted = user_data.sort_values('timestamp')
                
                # Look for privilege escalations
                prev_privilege = None
                for _, row in user_data_sorted.iterrows():
                    current_privilege = row[privilege_col]
                    
                    if prev_privilege and self._is_privilege_escalation(prev_privilege, current_privilege):
                        timestamp = row.get('timestamp', datetime.now())
                        
                        anomaly = BehavioralAnomaly(
                            timestamp=timestamp,
                            anomaly_type='privilege_escalation',
                            severity='high',
                            user_id=str(user_id),
                            session_id=row.get('session_id'),
                            description=f"Privilege escalation: {prev_privilege} → {current_privilege}",
                            confidence=0.8,
                            raw_data={
                                'previous_privilege': str(prev_privilege),
                                'new_privilege': str(current_privilege),
                                'escalation_detected': True
                            }
                        )
                        anomalies.append(anomaly)
                    
                    prev_privilege = current_privilege
        
        return anomalies
    
    def _detect_data_access_anomalies(self, df: pd.DataFrame) -> List[BehavioralAnomaly]:
        """Detect unusual data access patterns."""
        anomalies = []
        
        # Look for data access indicators
        data_columns = ['file_accessed', 'resource', 'data_type', 'table_name']
        data_col = None
        
        for col in data_columns:
            if col in df.columns:
                data_col = col
                break
        
        if not data_col or 'user_id' not in df.columns:
            return anomalies
        
        # Analyze data access patterns
        for user_id, user_data in df.groupby('user_id'):
            if len(user_data) < 5:
                continue
            
            # Count unique resources accessed
            unique_resources = user_data[data_col].nunique()
            total_accesses = len(user_data)
            
            # Detect users accessing many different resources (potential data harvesting)
            if unique_resources > 20 and unique_resources / total_accesses > 0.8:
                timestamp = user_data['timestamp'].min() if 'timestamp' in df.columns else datetime.now()
                
                severity = 'critical' if unique_resources > 50 else 'high'
                confidence = min(0.9, unique_resources / 100)
                
                anomaly = BehavioralAnomaly(
                    timestamp=timestamp,
                    anomaly_type='data_access_anomaly',
                    severity=severity,
                    user_id=str(user_id),
                    session_id=None,
                    description=f"Unusual data access pattern: {unique_resources} unique resources accessed",
                    confidence=confidence,
                    raw_data={
                        'unique_resources': int(unique_resources),
                        'total_accesses': int(total_accesses),
                        'diversity_ratio': float(unique_resources / total_accesses),
                        'resources_accessed': list(user_data[data_col].unique()[:10])  # First 10 for reference
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _is_privilege_escalation(self, prev_privilege: str, current_privilege: str) -> bool:
        """Determine if there's a privilege escalation."""
        # Define privilege hierarchy (lower number = higher privilege)
        privilege_levels = {
            'admin': 1,
            'administrator': 1,
            'root': 1,
            'manager': 2,
            'supervisor': 2,
            'user': 3,
            'guest': 4,
            'readonly': 5
        }
        
        prev_level = privilege_levels.get(prev_privilege.lower(), 3)
        current_level = privilege_levels.get(current_privilege.lower(), 3)
        
        return current_level < prev_level  # Lower number = higher privilege
    
    def get_behavioral_summary(self, anomalies: List[BehavioralAnomaly]) -> Dict[str, Any]:
        """Generate summary statistics for behavioral anomalies."""
        if not anomalies:
            return {'total': 0, 'by_type': {}, 'by_severity': {}}
        
        by_type = Counter(a.anomaly_type for a in anomalies)
        by_severity = Counter(a.severity for a in anomalies)
        
        # Count unique users affected
        unique_users = len(set(a.user_id for a in anomalies if a.user_id))
        
        return {
            'total': len(anomalies),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'unique_users_affected': unique_users,
            'high_confidence': len([a for a in anomalies if a.confidence > 0.7]),
            'time_range': {
                'start': min(a.timestamp for a in anomalies),
                'end': max(a.timestamp for a in anomalies)
            }
        }

