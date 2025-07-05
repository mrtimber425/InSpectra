"""
Network anomaly detection algorithms for CyberForensics Data Detective.
Detects various types of network security threats and suspicious activities.
Enhanced with ML ensemble for improved accuracy.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from utils.anomaly import Anomaly
from utils.helpers import calculate_velocity, detect_anomalous_amounts
from utils.ml_ensemble import MLEnsemble, AdvancedFeatureEngineering
from utils.behavioral_analysis import behavioral_engine
from config import config


class NetworkAnomalyDetector:
    """Advanced network anomaly detection system with ML ensemble."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML ensemble
        self.ml_ensemble = MLEnsemble(confidence_threshold=0.75)
        self.feature_engineer = AdvancedFeatureEngineering()
        self.is_ml_trained = False

        # Detection thresholds (configurable)
        self.thresholds = {
            'port_scan_threshold': config.get('network.port_scan_threshold', 10),
            'ddos_threshold': config.get('network.ddos_threshold', 1000),
            'data_exfil_threshold': config.get('network.data_exfil_threshold', 100 * 1024 * 1024),  # 100MB
            'lateral_movement_threshold': config.get('network.lateral_movement_threshold', 5),
            'dns_tunnel_threshold': config.get('network.dns_tunnel_threshold', 1000),
            'beacon_threshold': config.get('network.beacon_threshold', 0.8),
            'traffic_anomaly_threshold': config.get('network.traffic_anomaly_threshold', 3.0)
        }

        # Suspicious ports (commonly targeted)
        self.suspicious_ports = {
            22: 'SSH', 23: 'Telnet', 135: 'RPC', 139: 'NetBIOS', 445: 'SMB',
            1433: 'SQL Server', 1521: 'Oracle', 3389: 'RDP', 5432: 'PostgreSQL',
            5900: 'VNC', 6379: 'Redis', 27017: 'MongoDB'
        }

    def detect_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Main method to detect all types of network anomalies."""
        self.logger.info("Starting network anomaly detection")

        all_anomalies = []

        # Run all detection methods
        try:
            all_anomalies.extend(self.detect_port_scans(data))
            all_anomalies.extend(self.detect_ddos_attacks(data))
            all_anomalies.extend(self.detect_suspicious_connections(data))
            all_anomalies.extend(self.detect_data_exfiltration(data))
            all_anomalies.extend(self.detect_lateral_movement(data))
            all_anomalies.extend(self.detect_dns_tunneling(data))
            all_anomalies.extend(self.detect_beaconing_behavior(data))
            all_anomalies.extend(self.detect_unusual_traffic_patterns(data))
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")

        # Sort by severity and confidence
        all_anomalies.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
            x.confidence
        ), reverse=True)

        self.logger.info(f"Detected {len(all_anomalies)} network anomalies")
        return all_anomalies

    def detect_port_scans(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect port scanning activities."""
        anomalies = []

        # Required columns
        required_cols = ['source_ip']
        if 'dest_port' in data.columns:
            port_col = 'dest_port'
        elif 'destination_port' in data.columns:
            port_col = 'destination_port'
        elif 'port' in data.columns:
            port_col = 'port'
        else:
            self.logger.warning("No port column found for port scan detection")
            return anomalies

        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for port scan detection: {required_cols}")
            return anomalies

        try:
            # Group by source IP and count unique ports
            port_scans = data.groupby('source_ip')[port_col].nunique().reset_index()
            port_scans.columns = ['source_ip', 'unique_ports']

            # Filter for potential port scans
            threshold = self.thresholds['port_scan_threshold']
            suspicious_ips = port_scans[port_scans['unique_ports'] >= threshold]

            for _, row in suspicious_ips.iterrows():
                source_ip = row['source_ip']
                port_count = row['unique_ports']

                # Calculate severity based on port count
                if port_count >= 100:
                    severity = 'critical'
                    confidence = 0.95
                elif port_count >= 50:
                    severity = 'high'
                    confidence = 0.85
                elif port_count >= 20:
                    severity = 'medium'
                    confidence = 0.75
                else:
                    severity = 'low'
                    confidence = 0.65

                # Get timestamp of first scan
                ip_data = data[data['source_ip'] == source_ip]
                timestamp = ip_data['timestamp'].min() if 'timestamp' in data.columns else datetime.now()

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='port_scan',
                    description=f'Port scan detected from {source_ip} targeting {port_count} unique ports',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                        'source_ip': source_ip,
                        'unique_ports': int(port_count),
                        'threshold': threshold
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in port scan detection: {e}")

        return anomalies

    def detect_ddos_attacks(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect DDoS attack patterns."""
        anomalies = []

        if 'timestamp' not in data.columns or 'dest_ip' not in data.columns:
            self.logger.warning("Missing required columns for DDoS detection")
            return anomalies

        try:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Group by destination IP and time windows
            data['time_window'] = data['timestamp'].dt.floor('1min')

            ddos_candidates = data.groupby(['dest_ip', 'time_window']).size().reset_index(name='connection_count')

            # Filter for high connection counts
            threshold = self.thresholds['ddos_threshold']
            suspicious_targets = ddos_candidates[ddos_candidates['connection_count'] >= threshold]

            for _, row in suspicious_targets.iterrows():
                dest_ip = row['dest_ip']
                connection_count = row['connection_count']
                timestamp = row['time_window']

                # Calculate severity
                if connection_count >= 10000:
                    severity = 'critical'
                    confidence = 0.95
                elif connection_count >= 5000:
                    severity = 'high'
                    confidence = 0.85
                else:
                    severity = 'medium'
                    confidence = 0.75

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='ddos_attack',
                    description=f'Potential DDoS attack against {dest_ip} with {connection_count} connections in 1 minute',
                    severity=severity,
                    confidence=confidence,
                    raw_data={
                        'dest_ip': dest_ip,
                        'connection_count': int(connection_count),
                        'threshold': threshold,
                        'time_window': timestamp.isoformat()
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in DDoS detection: {e}")

        return anomalies

    def detect_suspicious_connections(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect connections to suspicious ports or IPs."""
        anomalies = []

        # Check for port column
        port_col = None
        for col in ['dest_port', 'destination_port', 'port']:
            if col in data.columns:
                port_col = col
                break

        if not port_col:
            self.logger.warning("No port column found for suspicious connection detection")
            return anomalies

        try:
            # Check for connections to suspicious ports
            suspicious_connections = data[data[port_col].isin(self.suspicious_ports.keys())]

            for _, row in suspicious_connections.iterrows():
                port = row[port_col]
                service = self.suspicious_ports.get(port, 'Unknown')

                timestamp = row.get('timestamp', datetime.now())
                source_ip = row.get('source_ip', 'Unknown')
                dest_ip = row.get('dest_ip', 'Unknown')

                # Higher severity for more dangerous ports
                if port in [22, 3389, 445]:  # SSH, RDP, SMB
                    severity = 'high'
                    confidence = 0.8
                elif port in [1433, 1521, 5432]:  # Database ports
                    severity = 'medium'
                    confidence = 0.7
                else:
                    severity = 'low'
                    confidence = 0.6

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='suspicious_connection',
                    description=f'Connection to suspicious port {port} ({service}) from {source_ip} to {dest_ip}',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                        'source_ip': source_ip,
                        'dest_ip': dest_ip,
                        'port': int(port),
                        'service': service
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in suspicious connection detection: {e}")

        return anomalies

    def detect_data_exfiltration(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect potential data exfiltration based on large data transfers."""
        anomalies = []

        # Check for bytes columns
        bytes_col = None
        for col in ['bytes_sent', 'bytes_out', 'bytes', 'size']:
            if col in data.columns:
                bytes_col = col
                break

        if not bytes_col:
            self.logger.warning("No bytes column found for data exfiltration detection")
            return anomalies

        try:
            # Filter for large transfers
            threshold = self.thresholds['data_exfil_threshold']
            large_transfers = data[data[bytes_col] >= threshold]

            for _, row in large_transfers.iterrows():
                bytes_transferred = row[bytes_col]
                timestamp = row.get('timestamp', datetime.now())
                source_ip = row.get('source_ip', 'Unknown')
                dest_ip = row.get('dest_ip', 'Unknown')

                # Calculate severity based on transfer size
                if bytes_transferred >= 1024 * 1024 * 1024:  # 1GB
                    severity = 'critical'
                    confidence = 0.9
                elif bytes_transferred >= 500 * 1024 * 1024:  # 500MB
                    severity = 'high'
                    confidence = 0.8
                else:
                    severity = 'medium'
                    confidence = 0.7

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='data_exfiltration',
                    description=f'Large data transfer detected: {bytes_transferred / (1024*1024):.1f} MB from {source_ip} to {dest_ip}',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                        'source_ip': source_ip,
                        'dest_ip': dest_ip,
                        'bytes_transferred': int(bytes_transferred),
                        'threshold': threshold
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in data exfiltration detection: {e}")

        return anomalies

    def detect_lateral_movement(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect lateral movement patterns in the network."""
        anomalies = []

        required_cols = ['source_ip', 'dest_ip']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning("Missing required columns for lateral movement detection")
            return anomalies

        try:
            # Look for internal-to-internal communications
            internal_data = data[
                data['source_ip'].str.startswith(('192.168.', '10.', '172.')) &
                data['dest_ip'].str.startswith(('192.168.', '10.', '172.'))
            ]

            if internal_data.empty:
                return anomalies

            # Count unique destinations per source
            lateral_movement = internal_data.groupby('source_ip')['dest_ip'].nunique().reset_index()
            lateral_movement.columns = ['source_ip', 'unique_destinations']

            # Filter for suspicious patterns
            threshold = self.thresholds['lateral_movement_threshold']
            suspicious_sources = lateral_movement[lateral_movement['unique_destinations'] >= threshold]

            for _, row in suspicious_sources.iterrows():
                source_ip = row['source_ip']
                dest_count = row['unique_destinations']

                # Get timestamp
                ip_data = data[data['source_ip'] == source_ip]
                timestamp = ip_data['timestamp'].min() if 'timestamp' in data.columns else datetime.now()

                # Calculate severity
                if dest_count >= 20:
                    severity = 'high'
                    confidence = 0.85
                elif dest_count >= 10:
                    severity = 'medium'
                    confidence = 0.75
                else:
                    severity = 'low'
                    confidence = 0.65

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='lateral_movement',
                    description=f'Potential lateral movement from {source_ip} to {dest_count} internal hosts',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                        'source_ip': source_ip,
                        'unique_destinations': int(dest_count),
                        'threshold': threshold
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in lateral movement detection: {e}")

        return anomalies

    def detect_dns_tunneling(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect DNS tunneling attempts."""
        anomalies = []

        # Look for DNS traffic (port 53) with unusual characteristics
        if 'dest_port' not in data.columns and 'port' not in data.columns:
            return anomalies

        try:
            # Filter DNS traffic
            port_col = 'dest_port' if 'dest_port' in data.columns else 'port'
            dns_data = data[data[port_col] == 53]

            if dns_data.empty:
                return anomalies

            # Look for unusual query sizes or frequencies
            if 'bytes' in dns_data.columns or 'size' in dns_data.columns:
                size_col = 'bytes' if 'bytes' in dns_data.columns else 'size'

                # Detect unusually large DNS queries
                large_queries = dns_data[dns_data[size_col] > self.thresholds['dns_tunnel_threshold']]

                for _, row in large_queries.iterrows():
                    timestamp = row.get('timestamp', datetime.now())
                    source_ip = row.get('source_ip', 'Unknown')
                    query_size = row[size_col]

                    anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='dns_tunneling',
                    description=f'Potential DNS tunneling detected from {source_ip} with query size {query_size} bytes',
                    severity='medium',
                    confidence=0.7,
                    source_data={
                            'source_ip': source_ip,
                            'query_size': int(query_size),
                            'threshold': self.thresholds['dns_tunnel_threshold']
                        }
                )
                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in DNS tunneling detection: {e}")

        return anomalies

    def detect_beaconing_behavior(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect beaconing behavior (regular communication patterns)."""
        anomalies = []

        required_cols = ['timestamp', 'source_ip', 'dest_ip']
        if not all(col in data.columns for col in required_cols):
            return anomalies

        try:
            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Group by source-destination pairs
            for (source_ip, dest_ip), group in data.groupby(['source_ip', 'dest_ip']):
                if len(group) < 5:  # Need at least 5 connections to detect pattern
                    continue

                # Calculate time intervals between connections
                group_sorted = group.sort_values('timestamp')
                time_diffs = group_sorted['timestamp'].diff().dt.total_seconds().dropna()

                if len(time_diffs) < 3:
                    continue

                # Check for regular intervals (low variance)
                mean_interval = time_diffs.mean()
                std_interval = time_diffs.std()

                if std_interval == 0:
                    regularity = 1.0
                else:
                    regularity = 1.0 - (std_interval / mean_interval)

                # Detect beaconing if intervals are very regular
                if regularity >= self.thresholds['beacon_threshold']:
                    timestamp = group_sorted['timestamp'].iloc[0]

                    # Higher confidence for more regular patterns
                    confidence = min(0.95, regularity)

                    if regularity >= 0.95:
                        severity = 'high'
                    elif regularity >= 0.85:
                        severity = 'medium'
                    else:
                        severity = 'low'

                    anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='beaconing',
                    description=f'Beaconing behavior detected from {source_ip} to {dest_ip} (regularity: {regularity:.2f})',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                            'source_ip': source_ip,
                            'dest_ip': dest_ip,
                            'regularity': regularity,
                            'mean_interval': mean_interval,
                            'connection_count': len(group)
                        }
                )
                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in beaconing detection: {e}")

        return anomalies

    def detect_unusual_traffic_patterns(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect unusual traffic patterns using statistical analysis."""
        anomalies = []

        # Check for bytes column
        bytes_col = None
        for col in ['bytes', 'bytes_sent', 'size']:
            if col in data.columns:
                bytes_col = col
                break

        if not bytes_col:
            return anomalies

        try:
            # Calculate statistical thresholds
            mean_bytes = data[bytes_col].mean()
            std_bytes = data[bytes_col].std()
            threshold = mean_bytes + (self.thresholds['traffic_anomaly_threshold'] * std_bytes)

            # Find outliers
            outliers = data[data[bytes_col] > threshold]

            for _, row in outliers.iterrows():
                timestamp = row.get('timestamp', datetime.now())
                source_ip = row.get('source_ip', 'Unknown')
                bytes_value = row[bytes_col]

                # Calculate z-score for severity
                z_score = (bytes_value - mean_bytes) / std_bytes if std_bytes > 0 else 0

                if z_score >= 5:
                    severity = 'high'
                    confidence = 0.8
                elif z_score >= 3:
                    severity = 'medium'
                    confidence = 0.7
                else:
                    severity = 'low'
                    confidence = 0.6

                anomaly = Anomaly(
                    timestamp=timestamp,
                    anomaly_type='traffic_anomaly',
                    description=f'Unusual traffic pattern detected from {source_ip}: {bytes_value} bytes (z-score: {z_score:.2f})',
                    severity=severity,
                    confidence=confidence,
                    source_data={
                        'source_ip': source_ip,
                        'bytes': int(bytes_value),
                        'z_score': z_score,
                        'threshold': threshold
                    }
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in traffic pattern detection: {e}")

        return anomalies


    def analyze_network_data(self, data: pd.DataFrame, selected_methods: List[str]) -> List[Anomaly]:
        """
        Analyze network data using selected detection methods enhanced with ML ensemble.
        This method is called by the main window.
        
        Args:
            data: Network data DataFrame
            selected_methods: List of detection methods to use
            
        Returns:
            List of detected anomalies with improved accuracy
        """
        self.logger.info(f"Analyzing network data with ML-enhanced methods: {selected_methods}")
        
        all_anomalies = []
        
        # Train ML ensemble if not already trained
        if not self.is_ml_trained and len(data) > 50:
            self.logger.info("Training ML ensemble on network data...")
            try:
                # Create network-specific features
                network_features = self.feature_engineer.create_network_features(data)
                enhanced_data = pd.concat([data, network_features], axis=1)
                
                # Define anomaly indicators for network data
                anomaly_indicators = ['bytes_out', 'packets_out', 'connections', 'port', 'protocol']
                
                # Train the ensemble
                self.is_ml_trained = self.ml_ensemble.train_ensemble(
                    enhanced_data, 
                    anomaly_indicators=anomaly_indicators
                )
                
                if self.is_ml_trained:
                    self.logger.info("ML ensemble training completed successfully")
                else:
                    self.logger.warning("ML ensemble training failed, using traditional methods")
                    
            except Exception as e:
                self.logger.error(f"ML training error: {e}")
                self.is_ml_trained = False
        
        # Map method names to actual detection functions
        method_mapping = {
            'port_scanning': self.detect_port_scans,
            'ddos_attacks': self.detect_ddos_attacks,
            'unusual_traffic': self.detect_unusual_traffic_patterns,
            'suspicious_connections': self.detect_suspicious_connections,
            'data_exfiltration': self.detect_data_exfiltration,
            'lateral_movement': self.detect_lateral_movement,
            'dns_tunneling': self.detect_dns_tunneling,
            'malware_beaconing': self.detect_beaconing_behavior
        }
        
        # Run selected detection methods
        for method_name in selected_methods:
            if method_name in method_mapping:
                try:
                    method = method_mapping[method_name]
                    anomalies = method(data)
                    all_anomalies.extend(anomalies)
                    self.logger.info(f"Method {method_name} found {len(anomalies)} anomalies")
                except Exception as e:
                    self.logger.error(f"Error in {method_name} detection: {e}")
        
        # Apply ML ensemble enhancement if trained
        if self.is_ml_trained and all_anomalies:
            try:
                self.logger.info("Applying ML ensemble enhancement...")
                enhanced_anomalies = self.enhance_anomalies_with_ml(data, all_anomalies)
                all_anomalies = enhanced_anomalies
            except Exception as e:
                self.logger.error(f"ML enhancement error: {e}")
        
        # Sort by severity and confidence
        all_anomalies.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
            x.confidence
        ), reverse=True)
        
        self.logger.info(f"Total anomalies detected: {len(all_anomalies)}")
        return all_anomalies
    
    def enhance_anomalies_with_ml(self, data: pd.DataFrame, anomalies: List[Anomaly]) -> List[Anomaly]:
        """
        Enhance detected anomalies using ML ensemble predictions
        """
        try:
            # Get ML predictions for the entire dataset
            ml_predictions, ml_confidences = self.ml_ensemble.predict_anomalies(data, return_probabilities=True)
            
            enhanced_anomalies = []
            
            for anomaly in anomalies:
                # Find corresponding data row (simplified approach)
                # In a real implementation, you'd want better row matching
                row_idx = 0
                if hasattr(anomaly, 'source_data') and isinstance(anomaly.source_data, dict):
                    # Try to find matching row based on source data
                    for idx, row in data.iterrows():
                        match_count = 0
                        for key, value in anomaly.source_data.items():
                            if key in data.columns and str(row[key]) == str(value):
                                match_count += 1
                        if match_count >= 2:  # At least 2 fields match
                            row_idx = idx
                            break
                
                # Get ML confidence for this row
                ml_confidence = ml_confidences[row_idx] if row_idx < len(ml_confidences) else 0.5
                ml_prediction = ml_predictions[row_idx] if row_idx < len(ml_predictions) else 0
                
                # Combine traditional and ML confidence
                combined_confidence = (anomaly.confidence + ml_confidence) / 2
                
                # Adjust severity based on ML prediction
                if ml_prediction == 1 and ml_confidence > 0.8:
                    # ML strongly agrees - increase severity
                    if anomaly.severity == 'low':
                        new_severity = 'medium'
                    elif anomaly.severity == 'medium':
                        new_severity = 'high'
                    elif anomaly.severity == 'high':
                        new_severity = 'critical'
                    else:
                        new_severity = anomaly.severity
                elif ml_prediction == 0 and ml_confidence > 0.8:
                    # ML disagrees - decrease severity
                    if anomaly.severity == 'critical':
                        new_severity = 'high'
                    elif anomaly.severity == 'high':
                        new_severity = 'medium'
                    elif anomaly.severity == 'medium':
                        new_severity = 'low'
                    else:
                        new_severity = anomaly.severity
                else:
                    new_severity = anomaly.severity
                
                # Create enhanced anomaly
                enhanced_anomaly = Anomaly(
                    anomaly_type=f"ML-Enhanced {anomaly.anomaly_type}",
                    description=f"{anomaly.description} (ML Confidence: {ml_confidence:.2f})",
                    severity=new_severity,
                    confidence=min(combined_confidence, 1.0),
                    timestamp=anomaly.timestamp,
                    source_data=anomaly.source_data
                )
                
                enhanced_anomalies.append(enhanced_anomaly)
            
            return enhanced_anomalies
            
        except Exception as e:
            self.logger.error(f"Error enhancing anomalies with ML: {e}")
            return anomalies
    
    def get_anomaly_summary(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """
        Generate summary statistics for detected anomalies.
        This method is called by the main window.
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Dictionary containing summary statistics
        """
        if not anomalies:
            return {
                'total': 0,
                'by_severity': {},
                'by_type': {},
                'avg_confidence': 0.0,
                'high_priority_count': 0,
                'recommendations': []
            }
        
        # Count by severity
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.anomaly_type
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(a.confidence for a in anomalies) / len(anomalies)
        
        # Count high priority anomalies
        high_priority_count = len([a for a in anomalies if a.is_high_priority()])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, severity_counts, type_counts)
        
        return {
            'total': len(anomalies),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'avg_confidence': avg_confidence,
            'high_priority_count': high_priority_count,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, anomalies: List[Anomaly], 
                                severity_counts: Dict[str, int], 
                                type_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations based on detected anomalies."""
        recommendations = []
        
        # High severity recommendations
        if severity_counts.get('critical', 0) > 0:
            recommendations.append("CRITICAL: Immediate investigation required for critical anomalies")
        
        if severity_counts.get('high', 0) > 0:
            recommendations.append("HIGH: Review high-severity anomalies within 1 hour")
        
        # Type-specific recommendations
        if type_counts.get('port_scan', 0) > 0:
            recommendations.append("Consider implementing port scan detection and blocking")
        
        if type_counts.get('ddos', 0) > 0:
            recommendations.append("Review DDoS protection mechanisms")
        
        if type_counts.get('data_exfiltration', 0) > 0:
            recommendations.append("Investigate potential data exfiltration - check data loss prevention")
        
        if type_counts.get('lateral_movement', 0) > 0:
            recommendations.append("Possible lateral movement detected - review network segmentation")
        
        if type_counts.get('suspicious_connections', 0) > 0:
            recommendations.append("Review firewall rules and network access controls")
        
        # General recommendations
        if len(anomalies) > 10:
            recommendations.append("High volume of anomalies detected - consider security audit")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no immediate action required")
        
        return recommendations

