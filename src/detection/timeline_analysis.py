"""
Timeline analysis module for CyberForensics Data Detective.
Creates and analyzes forensic timelines for digital investigations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.helpers import parse_timestamp


@dataclass
class TimelineEvent:
    """Represents an event in a forensic timeline."""
    timestamp: datetime
    event_type: str
    source: str
    description: str
    artifact: str
    user: Optional[str] = None
    host: Optional[str] = None
    file_path: Optional[str] = None
    process: Optional[str] = None
    network_info: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TimelineAnalyzer:
    """Analyzer for creating and analyzing forensic timelines."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Event type mappings
        self.event_type_mappings = {
            'file_creation': ['created', 'new file', 'file created'],
            'file_modification': ['modified', 'changed', 'updated', 'file modified'],
            'file_deletion': ['deleted', 'removed', 'file deleted'],
            'file_access': ['accessed', 'opened', 'read', 'file accessed'],
            'process_start': ['process started', 'execution', 'launched', 'started'],
            'process_end': ['process ended', 'terminated', 'killed', 'stopped'],
            'network_connection': ['connection', 'network', 'tcp', 'udp'],
            'login': ['login', 'logon', 'authentication', 'sign in'],
            'logout': ['logout', 'logoff', 'sign out'],
            'registry_change': ['registry', 'reg key', 'registry modified'],
            'service_start': ['service started', 'service running'],
            'service_stop': ['service stopped', 'service ended'],
            'email_sent': ['email sent', 'message sent', 'mail sent'],
            'email_received': ['email received', 'message received', 'mail received'],
            'web_access': ['web', 'http', 'browser', 'url accessed'],
            'usb_insertion': ['usb', 'removable media', 'device inserted'],
            'usb_removal': ['usb removed', 'device removed', 'media removed']
        }
    
    def create_timeline(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[TimelineEvent]:
        """Create a forensic timeline from data."""
        self.logger.info(f"Creating timeline from {len(df)} records")
        
        if timestamp_col not in df.columns:
            self.logger.error(f"Timestamp column '{timestamp_col}' not found")
            return []
        
        timeline_events = []
        
        for _, row in df.iterrows():
            event = self._create_timeline_event(row, timestamp_col)
            if event:
                timeline_events.append(event)
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Created timeline with {len(timeline_events)} events")
        return timeline_events
    
    def analyze_timeline_patterns(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze timeline for suspicious patterns."""
        self.logger.info(f"Analyzing timeline patterns for {len(events)} events")
        
        analysis_results = {
            'suspicious_sequences': self._find_suspicious_sequences(events),
            'time_gaps': self._find_time_gaps(events),
            'activity_clusters': self._find_activity_clusters(events),
            'user_activity_patterns': self._analyze_user_activity(events),
            'file_activity_patterns': self._analyze_file_activity(events),
            'network_activity_patterns': self._analyze_network_activity(events),
            'timeline_statistics': self._calculate_timeline_statistics(events)
        }
        
        return analysis_results
    
    def find_events_around_time(self, events: List[TimelineEvent], 
                               target_time: datetime, 
                               window_minutes: int = 30) -> List[TimelineEvent]:
        """Find events around a specific time."""
        start_time = target_time - timedelta(minutes=window_minutes)
        end_time = target_time + timedelta(minutes=window_minutes)
        
        return [event for event in events 
                if start_time <= event.timestamp <= end_time]
    
    def filter_events_by_type(self, events: List[TimelineEvent], 
                             event_types: List[str]) -> List[TimelineEvent]:
        """Filter events by type."""
        return [event for event in events if event.event_type in event_types]
    
    def filter_events_by_user(self, events: List[TimelineEvent], 
                             users: List[str]) -> List[TimelineEvent]:
        """Filter events by user."""
        return [event for event in events 
                if event.user and event.user in users]
    
    def create_timeline_report(self, events: List[TimelineEvent], 
                              analysis: Dict[str, Any]) -> str:
        """Create a comprehensive timeline report."""
        report_lines = []
        
        # Header
        report_lines.append("DIGITAL FORENSICS TIMELINE ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Events: {len(events)}")
        
        if events:
            report_lines.append(f"Time Range: {events[0].timestamp} to {events[-1].timestamp}")
        
        report_lines.append("")
        
        # Statistics
        stats = analysis.get('timeline_statistics', {})
        report_lines.append("TIMELINE STATISTICS")
        report_lines.append("-" * 20)
        
        for key, value in stats.items():
            report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        report_lines.append("")
        
        # Suspicious sequences
        suspicious = analysis.get('suspicious_sequences', [])
        if suspicious:
            report_lines.append("SUSPICIOUS ACTIVITY SEQUENCES")
            report_lines.append("-" * 30)
            
            for i, sequence in enumerate(suspicious[:10], 1):  # Top 10
                report_lines.append(f"{i}. {sequence['description']}")
                report_lines.append(f"   Time: {sequence['start_time']} - {sequence['end_time']}")
                report_lines.append(f"   Events: {sequence['event_count']}")
                report_lines.append("")
        
        # Time gaps
        gaps = analysis.get('time_gaps', [])
        if gaps:
            report_lines.append("SIGNIFICANT TIME GAPS")
            report_lines.append("-" * 20)
            
            for gap in gaps[:5]:  # Top 5 gaps
                report_lines.append(f"Gap: {gap['duration_hours']:.1f} hours")
                report_lines.append(f"From: {gap['start_time']} to {gap['end_time']}")
                report_lines.append("")
        
        # Activity clusters
        clusters = analysis.get('activity_clusters', [])
        if clusters:
            report_lines.append("HIGH ACTIVITY PERIODS")
            report_lines.append("-" * 20)
            
            for cluster in clusters[:5]:  # Top 5 clusters
                report_lines.append(f"Period: {cluster['start_time']} - {cluster['end_time']}")
                report_lines.append(f"Events: {cluster['event_count']}")
                report_lines.append(f"Primary Activity: {cluster['primary_activity']}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _create_timeline_event(self, row: pd.Series, timestamp_col: str) -> Optional[TimelineEvent]:
        """Create a timeline event from a data row."""
        try:
            timestamp = row[timestamp_col]
            if pd.isna(timestamp):
                return None
            
            # Ensure timestamp is datetime
            if not isinstance(timestamp, datetime):
                timestamp = pd.to_datetime(timestamp)
            
            # Determine event type
            event_type = self._determine_event_type(row)
            
            # Extract relevant information
            description = self._create_event_description(row, event_type)
            source = self._determine_source(row)
            artifact = self._determine_artifact(row)
            
            # Extract optional fields
            user = row.get('user', row.get('user_id', row.get('username')))
            host = row.get('host', row.get('hostname', row.get('computer')))
            file_path = row.get('file_path', row.get('path', row.get('filename')))
            process = row.get('process', row.get('process_name', row.get('executable')))
            network_info = self._extract_network_info(row)
            
            # Create metadata
            metadata = {
                'original_data': row.to_dict(),
                'confidence': self._calculate_event_confidence(row, event_type)
            }
            
            return TimelineEvent(
                timestamp=timestamp,
                event_type=event_type,
                source=source,
                description=description,
                artifact=artifact,
                user=str(user) if pd.notna(user) else None,
                host=str(host) if pd.notna(host) else None,
                file_path=str(file_path) if pd.notna(file_path) else None,
                process=str(process) if pd.notna(process) else None,
                network_info=network_info,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating timeline event: {e}")
            return None
    
    def _determine_event_type(self, row: pd.Series) -> str:
        """Determine the event type from row data."""
        # Check for explicit event type column
        for col in ['event_type', 'type', 'action', 'activity']:
            if col in row.index and pd.notna(row[col]):
                event_value = str(row[col]).lower()
                
                # Map to standard event types
                for event_type, keywords in self.event_type_mappings.items():
                    if any(keyword in event_value for keyword in keywords):
                        return event_type
                
                return event_value
        
        # Infer from other columns
        if any(col in row.index for col in ['file_path', 'filename', 'path']):
            if any(keyword in str(row).lower() for keyword in ['created', 'new']):
                return 'file_creation'
            elif any(keyword in str(row).lower() for keyword in ['modified', 'changed']):
                return 'file_modification'
            elif any(keyword in str(row).lower() for keyword in ['deleted', 'removed']):
                return 'file_deletion'
            else:
                return 'file_access'
        
        if any(col in row.index for col in ['process', 'process_name', 'executable']):
            if any(keyword in str(row).lower() for keyword in ['started', 'launched']):
                return 'process_start'
            elif any(keyword in str(row).lower() for keyword in ['ended', 'terminated']):
                return 'process_end'
            else:
                return 'process_activity'
        
        if any(col in row.index for col in ['source_ip', 'dest_ip', 'network']):
            return 'network_connection'
        
        if any(keyword in str(row).lower() for keyword in ['login', 'logon', 'authentication']):
            return 'login'
        
        if any(keyword in str(row).lower() for keyword in ['logout', 'logoff']):
            return 'logout'
        
        return 'unknown'
    
    def _create_event_description(self, row: pd.Series, event_type: str) -> str:
        """Create a human-readable event description."""
        # Try to find a description column
        for col in ['description', 'message', 'details', 'summary']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        
        # Generate description based on event type and available data
        if event_type.startswith('file_'):
            file_path = row.get('file_path', row.get('filename', row.get('path', 'unknown file')))
            action = event_type.replace('file_', '').replace('_', ' ')
            return f"File {action}: {file_path}"
        
        elif event_type.startswith('process_'):
            process = row.get('process', row.get('process_name', row.get('executable', 'unknown process')))
            action = event_type.replace('process_', '').replace('_', ' ')
            return f"Process {action}: {process}"
        
        elif event_type == 'network_connection':
            source_ip = row.get('source_ip', 'unknown')
            dest_ip = row.get('dest_ip', row.get('destination_ip', 'unknown'))
            port = row.get('port', row.get('dest_port', ''))
            return f"Network connection: {source_ip} → {dest_ip}:{port}"
        
        elif event_type in ['login', 'logout']:
            user = row.get('user', row.get('username', 'unknown user'))
            return f"User {event_type}: {user}"
        
        else:
            # Generic description
            return f"{event_type.replace('_', ' ').title()}"
    
    def _determine_source(self, row: pd.Series) -> str:
        """Determine the data source."""
        for col in ['source', 'log_source', 'data_source']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        
        # Infer from column names or data
        if any(col in row.index for col in ['file_path', 'filename']):
            return 'file_system'
        elif any(col in row.index for col in ['process', 'process_name']):
            return 'process_monitor'
        elif any(col in row.index for col in ['source_ip', 'network']):
            return 'network_monitor'
        elif any(col in row.index for col in ['user', 'username']):
            return 'security_log'
        else:
            return 'unknown'
    
    def _determine_artifact(self, row: pd.Series) -> str:
        """Determine the digital artifact."""
        # File artifacts
        if any(col in row.index for col in ['file_path', 'filename', 'path']):
            return row.get('file_path', row.get('filename', row.get('path', 'file')))
        
        # Process artifacts
        if any(col in row.index for col in ['process', 'process_name', 'executable']):
            return row.get('process', row.get('process_name', row.get('executable', 'process')))
        
        # Network artifacts
        if any(col in row.index for col in ['source_ip', 'dest_ip']):
            source = row.get('source_ip', '')
            dest = row.get('dest_ip', row.get('destination_ip', ''))
            return f"{source}→{dest}" if source and dest else source or dest
        
        # Registry artifacts
        if any(col in row.index for col in ['registry_key', 'reg_key']):
            return row.get('registry_key', row.get('reg_key', 'registry'))
        
        return 'system'
    
    def _extract_network_info(self, row: pd.Series) -> Optional[str]:
        """Extract network information."""
        network_fields = []
        
        if 'source_ip' in row.index and pd.notna(row['source_ip']):
            network_fields.append(f"src:{row['source_ip']}")
        
        if 'dest_ip' in row.index and pd.notna(row['dest_ip']):
            network_fields.append(f"dst:{row['dest_ip']}")
        elif 'destination_ip' in row.index and pd.notna(row['destination_ip']):
            network_fields.append(f"dst:{row['destination_ip']}")
        
        if 'port' in row.index and pd.notna(row['port']):
            network_fields.append(f"port:{row['port']}")
        elif 'dest_port' in row.index and pd.notna(row['dest_port']):
            network_fields.append(f"port:{row['dest_port']}")
        
        if 'protocol' in row.index and pd.notna(row['protocol']):
            network_fields.append(f"proto:{row['protocol']}")
        
        return " ".join(network_fields) if network_fields else None
    
    def _calculate_event_confidence(self, row: pd.Series, event_type: str) -> float:
        """Calculate confidence score for the event."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if any(col in row.index for col in ['timestamp', 'time']):
            confidence += 0.2
        
        if any(col in row.index for col in ['user', 'username', 'user_id']):
            confidence += 0.1
        
        if any(col in row.index for col in ['host', 'hostname', 'computer']):
            confidence += 0.1
        
        if event_type != 'unknown':
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _find_suspicious_sequences(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Find suspicious sequences of events."""
        suspicious_sequences = []
        
        # Define suspicious patterns
        suspicious_patterns = [
            {
                'name': 'Rapid file deletion',
                'events': ['file_deletion'],
                'min_count': 5,
                'max_duration_minutes': 10
            },
            {
                'name': 'Mass file access',
                'events': ['file_access'],
                'min_count': 20,
                'max_duration_minutes': 30
            },
            {
                'name': 'Process injection sequence',
                'events': ['process_start', 'process_end'],
                'min_count': 3,
                'max_duration_minutes': 5
            },
            {
                'name': 'Login brute force',
                'events': ['login'],
                'min_count': 10,
                'max_duration_minutes': 15
            }
        ]
        
        for pattern in suspicious_patterns:
            sequences = self._find_pattern_sequences(events, pattern)
            suspicious_sequences.extend(sequences)
        
        return suspicious_sequences
    
    def _find_pattern_sequences(self, events: List[TimelineEvent], pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find sequences matching a specific pattern."""
        sequences = []
        pattern_events = [e for e in events if e.event_type in pattern['events']]
        
        if len(pattern_events) < pattern['min_count']:
            return sequences
        
        # Group events by time windows
        window_duration = timedelta(minutes=pattern['max_duration_minutes'])
        
        i = 0
        while i < len(pattern_events):
            window_start = pattern_events[i].timestamp
            window_end = window_start + window_duration
            
            # Count events in this window
            window_events = []
            j = i
            while j < len(pattern_events) and pattern_events[j].timestamp <= window_end:
                window_events.append(pattern_events[j])
                j += 1
            
            if len(window_events) >= pattern['min_count']:
                sequences.append({
                    'pattern_name': pattern['name'],
                    'description': f"{pattern['name']}: {len(window_events)} events",
                    'start_time': window_start,
                    'end_time': window_events[-1].timestamp,
                    'event_count': len(window_events),
                    'events': window_events
                })
            
            i = j if j > i else i + 1
        
        return sequences
    
    def _find_time_gaps(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Find significant time gaps in the timeline."""
        if len(events) < 2:
            return []
        
        gaps = []
        
        for i in range(1, len(events)):
            time_diff = events[i].timestamp - events[i-1].timestamp
            hours_diff = time_diff.total_seconds() / 3600
            
            # Consider gaps > 2 hours as significant
            if hours_diff > 2:
                gaps.append({
                    'start_time': events[i-1].timestamp,
                    'end_time': events[i].timestamp,
                    'duration_hours': hours_diff,
                    'previous_event': events[i-1].description,
                    'next_event': events[i].description
                })
        
        # Sort by duration (longest first)
        gaps.sort(key=lambda x: x['duration_hours'], reverse=True)
        
        return gaps
    
    def _find_activity_clusters(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Find periods of high activity."""
        if len(events) < 10:
            return []
        
        # Group events by hour
        hourly_counts = {}
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_counts:
                hourly_counts[hour_key] = []
            hourly_counts[hour_key].append(event)
        
        # Find hours with high activity (> 75th percentile)
        counts = [len(events) for events in hourly_counts.values()]
        if not counts:
            return []
        
        threshold = np.percentile(counts, 75)
        
        clusters = []
        for hour, hour_events in hourly_counts.items():
            if len(hour_events) > threshold:
                # Determine primary activity type
                event_types = [e.event_type for e in hour_events]
                primary_activity = max(set(event_types), key=event_types.count)
                
                clusters.append({
                    'start_time': hour,
                    'end_time': hour + timedelta(hours=1),
                    'event_count': len(hour_events),
                    'primary_activity': primary_activity,
                    'event_types': list(set(event_types)),
                    'events': hour_events
                })
        
        # Sort by event count (highest first)
        clusters.sort(key=lambda x: x['event_count'], reverse=True)
        
        return clusters
    
    def _analyze_user_activity(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        user_stats = {}
        
        for event in events:
            if event.user:
                if event.user not in user_stats:
                    user_stats[event.user] = {
                        'total_events': 0,
                        'event_types': set(),
                        'first_activity': event.timestamp,
                        'last_activity': event.timestamp,
                        'hosts': set()
                    }
                
                stats = user_stats[event.user]
                stats['total_events'] += 1
                stats['event_types'].add(event.event_type)
                stats['first_activity'] = min(stats['first_activity'], event.timestamp)
                stats['last_activity'] = max(stats['last_activity'], event.timestamp)
                
                if event.host:
                    stats['hosts'].add(event.host)
        
        # Convert sets to lists for JSON serialization
        for user, stats in user_stats.items():
            stats['event_types'] = list(stats['event_types'])
            stats['hosts'] = list(stats['hosts'])
            stats['activity_duration_hours'] = (stats['last_activity'] - stats['first_activity']).total_seconds() / 3600
        
        return user_stats
    
    def _analyze_file_activity(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze file activity patterns."""
        file_events = [e for e in events if e.event_type.startswith('file_')]
        
        if not file_events:
            return {}
        
        file_stats = {
            'total_file_events': len(file_events),
            'file_creations': len([e for e in file_events if e.event_type == 'file_creation']),
            'file_modifications': len([e for e in file_events if e.event_type == 'file_modification']),
            'file_deletions': len([e for e in file_events if e.event_type == 'file_deletion']),
            'file_accesses': len([e for e in file_events if e.event_type == 'file_access']),
            'unique_files': len(set(e.file_path for e in file_events if e.file_path)),
            'most_active_directories': self._get_most_active_directories(file_events)
        }
        
        return file_stats
    
    def _analyze_network_activity(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze network activity patterns."""
        network_events = [e for e in events if e.event_type == 'network_connection']
        
        if not network_events:
            return {}
        
        network_stats = {
            'total_connections': len(network_events),
            'unique_sources': len(set(e.network_info.split()[0].split(':')[1] if e.network_info and 'src:' in e.network_info else None for e in network_events)),
            'unique_destinations': len(set(e.network_info.split()[1].split(':')[1] if e.network_info and 'dst:' in e.network_info else None for e in network_events)),
            'connection_timeline': [(e.timestamp, e.network_info) for e in network_events[:10]]  # First 10 for sample
        }
        
        return network_stats
    
    def _get_most_active_directories(self, file_events: List[TimelineEvent]) -> List[Tuple[str, int]]:
        """Get directories with most file activity."""
        directory_counts = {}
        
        for event in file_events:
            if event.file_path:
                # Extract directory from file path
                if '/' in event.file_path:
                    directory = '/'.join(event.file_path.split('/')[:-1])
                elif '\\' in event.file_path:
                    directory = '\\'.join(event.file_path.split('\\')[:-1])
                else:
                    directory = 'root'
                
                directory_counts[directory] = directory_counts.get(directory, 0) + 1
        
        # Return top 10 directories
        return sorted(directory_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _calculate_timeline_statistics(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Calculate overall timeline statistics."""
        if not events:
            return {}
        
        event_types = [e.event_type for e in events]
        users = [e.user for e in events if e.user]
        hosts = [e.host for e in events if e.host]
        
        return {
            'total_events': len(events),
            'unique_event_types': len(set(event_types)),
            'unique_users': len(set(users)),
            'unique_hosts': len(set(hosts)),
            'time_span_hours': (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600,
            'events_per_hour': len(events) / max(1, (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600),
            'most_common_event_type': max(set(event_types), key=event_types.count) if event_types else 'none',
            'most_active_user': max(set(users), key=users.count) if users else 'none',
            'most_active_host': max(set(hosts), key=hosts.count) if hosts else 'none'
        }

