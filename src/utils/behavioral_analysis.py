"""
Smart Behavioral Analysis Module for Forensic Investigation
Implements advanced pattern recognition and behavioral anomaly detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class BehavioralPattern:
    """Represents a detected behavioral pattern"""
    
    def __init__(self, pattern_type: str, description: str, confidence: float, 
                 severity: str, entities: List[str], metrics: Dict[str, Any]):
        self.pattern_type = pattern_type
        self.description = description
        self.confidence = confidence
        self.severity = severity
        self.entities = entities  # Users, IPs, accounts involved
        self.metrics = metrics
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            'pattern_type': self.pattern_type,
            'description': self.description,
            'confidence': self.confidence,
            'severity': self.severity,
            'entities': self.entities,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }

class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for anomaly detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.user_profiles = {}
        self.baseline_established = False
    
    def analyze_user_behavior(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Analyze user behavior patterns in the data"""
        patterns = []
        
        # Identify user columns
        user_columns = self._identify_user_columns(data)
        if not user_columns:
            return patterns
        
        user_col = user_columns[0]
        
        # Build user profiles
        self._build_user_profiles(data, user_col)
        
        # Detect behavioral anomalies
        patterns.extend(self._detect_activity_patterns(data, user_col))
        patterns.extend(self._detect_temporal_anomalies(data, user_col))
        patterns.extend(self._detect_volume_anomalies(data, user_col))
        patterns.extend(self._detect_location_anomalies(data, user_col))
        
        return patterns
    
    def _identify_user_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns that likely contain user identifiers"""
        user_keywords = ['user', 'account', 'customer', 'client', 'id']
        user_columns = []
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in user_keywords):
                # Check if column has reasonable cardinality for users
                unique_count = data[col].nunique()
                if 2 <= unique_count <= len(data) * 0.8:  # Not too few, not too many
                    user_columns.append(col)
        
        return user_columns
    
    def _build_user_profiles(self, data: pd.DataFrame, user_col: str):
        """Build behavioral profiles for each user"""
        self.user_profiles = {}
        
        for user in data[user_col].unique():
            user_data = data[data[user_col] == user]
            
            profile = {
                'total_activities': len(user_data),
                'unique_days': self._get_unique_days(user_data),
                'activity_hours': self._get_activity_hours(user_data),
                'activity_patterns': self._get_activity_patterns(user_data),
                'volume_metrics': self._get_volume_metrics(user_data),
                'location_patterns': self._get_location_patterns(user_data)
            }
            
            self.user_profiles[user] = profile
        
        self.baseline_established = True
    
    def _get_unique_days(self, user_data: pd.DataFrame) -> int:
        """Get number of unique days user was active"""
        time_cols = [col for col in user_data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            try:
                dates = pd.to_datetime(user_data[time_cols[0]])
                return dates.dt.date.nunique()
            except:
                pass
        return 1
    
    def _get_activity_hours(self, user_data: pd.DataFrame) -> List[int]:
        """Get hours when user is typically active"""
        time_cols = [col for col in user_data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            try:
                dates = pd.to_datetime(user_data[time_cols[0]])
                return dates.dt.hour.tolist()
            except:
                pass
        return []
    
    def _get_activity_patterns(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze activity patterns"""
        patterns = {
            'avg_daily_activities': len(user_data) / max(1, self._get_unique_days(user_data)),
            'activity_variance': 0,
            'peak_hours': [],
            'weekend_activity': 0
        }
        
        time_cols = [col for col in user_data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            try:
                dates = pd.to_datetime(user_data[time_cols[0]])
                
                # Peak hours
                hour_counts = dates.dt.hour.value_counts()
                patterns['peak_hours'] = hour_counts.head(3).index.tolist()
                
                # Weekend activity
                weekend_mask = dates.dt.dayofweek >= 5
                patterns['weekend_activity'] = weekend_mask.sum() / len(dates)
                
                # Activity variance
                daily_counts = dates.dt.date.value_counts()
                patterns['activity_variance'] = daily_counts.std() if len(daily_counts) > 1 else 0
                
            except:
                pass
        
        return patterns
    
    def _get_volume_metrics(self, user_data: pd.DataFrame) -> Dict[str, float]:
        """Get volume-related metrics"""
        metrics = {}
        
        # Look for amount/volume columns
        amount_cols = [col for col in user_data.columns 
                      if any(keyword in col.lower() for keyword in ['amount', 'value', 'size', 'bytes'])]
        
        for col in amount_cols:
            if user_data[col].dtype in ['int64', 'float64']:
                metrics[f'{col}_total'] = user_data[col].sum()
                metrics[f'{col}_avg'] = user_data[col].mean()
                metrics[f'{col}_max'] = user_data[col].max()
                metrics[f'{col}_std'] = user_data[col].std()
        
        return metrics
    
    def _get_location_patterns(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze location-based patterns"""
        patterns = {}
        
        # Look for location columns
        location_cols = [col for col in user_data.columns 
                        if any(keyword in col.lower() for keyword in ['ip', 'location', 'country', 'city'])]
        
        for col in location_cols:
            unique_locations = user_data[col].nunique()
            most_common = user_data[col].mode().iloc[0] if len(user_data[col].mode()) > 0 else None
            
            patterns[f'{col}_unique_count'] = unique_locations
            patterns[f'{col}_most_common'] = most_common
            patterns[f'{col}_diversity'] = unique_locations / len(user_data)
        
        return patterns
    
    def _detect_activity_patterns(self, data: pd.DataFrame, user_col: str) -> List[BehavioralPattern]:
        """Detect unusual activity patterns"""
        patterns = []
        
        if not self.baseline_established:
            return patterns
        
        # Calculate baseline metrics
        all_activities = [profile['total_activities'] for profile in self.user_profiles.values()]
        avg_activities = np.mean(all_activities)
        std_activities = np.std(all_activities)
        
        for user, profile in self.user_profiles.items():
            # Detect hyperactive users
            if profile['total_activities'] > avg_activities + 3 * std_activities:
                patterns.append(BehavioralPattern(
                    pattern_type="Hyperactive User",
                    description=f"User {user} shows unusually high activity ({profile['total_activities']} vs avg {avg_activities:.1f})",
                    confidence=min(0.9, (profile['total_activities'] - avg_activities) / (3 * std_activities)),
                    severity="medium" if profile['total_activities'] > avg_activities + 2 * std_activities else "low",
                    entities=[user],
                    metrics={'activity_count': profile['total_activities'], 'baseline_avg': avg_activities}
                ))
            
            # Detect inactive users
            elif profile['total_activities'] < avg_activities - 2 * std_activities and profile['total_activities'] > 0:
                patterns.append(BehavioralPattern(
                    pattern_type="Inactive User",
                    description=f"User {user} shows unusually low activity ({profile['total_activities']} vs avg {avg_activities:.1f})",
                    confidence=0.7,
                    severity="low",
                    entities=[user],
                    metrics={'activity_count': profile['total_activities'], 'baseline_avg': avg_activities}
                ))
        
        return patterns
    
    def _detect_temporal_anomalies(self, data: pd.DataFrame, user_col: str) -> List[BehavioralPattern]:
        """Detect temporal behavior anomalies"""
        patterns = []
        
        for user, profile in self.user_profiles.items():
            activity_hours = profile['activity_hours']
            if not activity_hours:
                continue
            
            # Detect unusual time patterns
            hour_counts = Counter(activity_hours)
            
            # Night activity (11 PM - 5 AM)
            night_hours = [23, 0, 1, 2, 3, 4, 5]
            night_activity = sum(hour_counts.get(hour, 0) for hour in night_hours)
            total_activity = sum(hour_counts.values())
            
            if night_activity / total_activity > 0.3:  # More than 30% night activity
                patterns.append(BehavioralPattern(
                    pattern_type="Unusual Time Activity",
                    description=f"User {user} shows high night-time activity ({night_activity}/{total_activity} activities)",
                    confidence=0.8,
                    severity="medium",
                    entities=[user],
                    metrics={'night_activity_ratio': night_activity / total_activity}
                ))
            
            # Weekend activity anomalies
            weekend_ratio = profile['activity_patterns'].get('weekend_activity', 0)
            if weekend_ratio > 0.7:  # More than 70% weekend activity
                patterns.append(BehavioralPattern(
                    pattern_type="Weekend Activity Anomaly",
                    description=f"User {user} shows predominantly weekend activity ({weekend_ratio:.1%})",
                    confidence=0.7,
                    severity="low",
                    entities=[user],
                    metrics={'weekend_activity_ratio': weekend_ratio}
                ))
        
        return patterns
    
    def _detect_volume_anomalies(self, data: pd.DataFrame, user_col: str) -> List[BehavioralPattern]:
        """Detect volume-based anomalies"""
        patterns = []
        
        # Collect all volume metrics
        all_metrics = defaultdict(list)
        for profile in self.user_profiles.values():
            for metric, value in profile['volume_metrics'].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    all_metrics[metric].append(value)
        
        # Detect outliers for each metric
        for metric, values in all_metrics.items():
            if len(values) < 3:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            for user, profile in self.user_profiles.items():
                user_value = profile['volume_metrics'].get(metric, 0)
                
                if user_value > mean_val + 3 * std_val:
                    patterns.append(BehavioralPattern(
                        pattern_type="Volume Anomaly",
                        description=f"User {user} shows unusual {metric}: {user_value:.2f} vs avg {mean_val:.2f}",
                        confidence=0.8,
                        severity="high" if user_value > mean_val + 4 * std_val else "medium",
                        entities=[user],
                        metrics={metric: user_value, f'{metric}_baseline': mean_val}
                    ))
        
        return patterns
    
    def _detect_location_anomalies(self, data: pd.DataFrame, user_col: str) -> List[BehavioralPattern]:
        """Detect location-based anomalies"""
        patterns = []
        
        for user, profile in self.user_profiles.items():
            location_patterns = profile['location_patterns']
            
            for pattern_key, value in location_patterns.items():
                if 'diversity' in pattern_key and isinstance(value, float):
                    # High location diversity might indicate account sharing or compromise
                    if value > 0.5:  # More than 50% unique locations
                        patterns.append(BehavioralPattern(
                            pattern_type="Location Diversity Anomaly",
                            description=f"User {user} shows high location diversity ({value:.1%})",
                            confidence=0.7,
                            severity="medium",
                            entities=[user],
                            metrics={'location_diversity': value}
                        ))
        
        return patterns

class NetworkBehaviorAnalyzer:
    """Analyzes network behavior patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_network_behavior(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Analyze network behavior patterns"""
        patterns = []
        
        patterns.extend(self._detect_communication_patterns(data))
        patterns.extend(self._detect_traffic_anomalies(data))
        patterns.extend(self._detect_protocol_anomalies(data))
        patterns.extend(self._detect_timing_patterns(data))
        
        return patterns
    
    def _detect_communication_patterns(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect unusual communication patterns"""
        patterns = []
        
        # Look for IP columns
        ip_cols = [col for col in data.columns if 'ip' in col.lower()]
        if len(ip_cols) >= 2:
            src_ip_col = ip_cols[0]
            dst_ip_col = ip_cols[1]
            
            # Detect one-to-many communication (potential C&C)
            src_comm_counts = data.groupby(src_ip_col)[dst_ip_col].nunique()
            high_comm_threshold = src_comm_counts.quantile(0.95)
            
            for src_ip, dst_count in src_comm_counts.items():
                if dst_count > high_comm_threshold and dst_count > 10:
                    patterns.append(BehavioralPattern(
                        pattern_type="High Fan-out Communication",
                        description=f"IP {src_ip} communicates with {dst_count} unique destinations",
                        confidence=0.8,
                        severity="high" if dst_count > high_comm_threshold * 1.5 else "medium",
                        entities=[src_ip],
                        metrics={'destination_count': dst_count, 'threshold': high_comm_threshold}
                    ))
            
            # Detect many-to-one communication (potential data collection)
            dst_comm_counts = data.groupby(dst_ip_col)[src_ip_col].nunique()
            high_dst_threshold = dst_comm_counts.quantile(0.95)
            
            for dst_ip, src_count in dst_comm_counts.items():
                if src_count > high_dst_threshold and src_count > 10:
                    patterns.append(BehavioralPattern(
                        pattern_type="High Fan-in Communication",
                        description=f"IP {dst_ip} receives communication from {src_count} unique sources",
                        confidence=0.8,
                        severity="high" if src_count > high_dst_threshold * 1.5 else "medium",
                        entities=[dst_ip],
                        metrics={'source_count': src_count, 'threshold': high_dst_threshold}
                    ))
        
        return patterns
    
    def _detect_traffic_anomalies(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect traffic volume anomalies"""
        patterns = []
        
        # Look for traffic volume columns
        volume_cols = [col for col in data.columns 
                      if any(keyword in col.lower() for keyword in ['bytes', 'size', 'length', 'packets'])]
        
        for vol_col in volume_cols:
            if data[vol_col].dtype in ['int64', 'float64']:
                # Detect outliers using IQR method
                Q1 = data[vol_col].quantile(0.25)
                Q3 = data[vol_col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 3 * IQR
                
                outliers = data[data[vol_col] > upper_bound]
                
                if len(outliers) > 0:
                    # Group outliers by source IP if available
                    ip_cols = [col for col in data.columns if 'ip' in col.lower()]
                    if ip_cols:
                        ip_col = ip_cols[0]
                        outlier_ips = outliers[ip_col].value_counts()
                        
                        for ip, count in outlier_ips.items():
                            if count > 1:  # Multiple high-volume transfers
                                patterns.append(BehavioralPattern(
                                    pattern_type="High Volume Traffic",
                                    description=f"IP {ip} generated {count} high-volume transfers in {vol_col}",
                                    confidence=0.9,
                                    severity="high",
                                    entities=[ip],
                                    metrics={'outlier_count': count, 'volume_column': vol_col}
                                ))
        
        return patterns
    
    def _detect_protocol_anomalies(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect protocol usage anomalies"""
        patterns = []
        
        # Look for protocol columns
        protocol_cols = [col for col in data.columns if 'protocol' in col.lower()]
        
        for proto_col in protocol_cols:
            protocol_counts = data[proto_col].value_counts()
            total_connections = len(data)
            
            # Detect unusual protocol usage
            for protocol, count in protocol_counts.items():
                usage_ratio = count / total_connections
                
                # Flag protocols with very low usage but present
                if 0.001 < usage_ratio < 0.01:  # Between 0.1% and 1%
                    patterns.append(BehavioralPattern(
                        pattern_type="Unusual Protocol Usage",
                        description=f"Rare protocol {protocol} used in {count} connections ({usage_ratio:.1%})",
                        confidence=0.6,
                        severity="low",
                        entities=[protocol],
                        metrics={'usage_count': count, 'usage_ratio': usage_ratio}
                    ))
        
        return patterns
    
    def _detect_timing_patterns(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect timing-based patterns"""
        patterns = []
        
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if time_cols:
            time_col = time_cols[0]
            try:
                timestamps = pd.to_datetime(data[time_col])
                
                # Detect regular intervals (potential beaconing)
                if len(timestamps) > 10:
                    time_diffs = timestamps.diff().dropna()
                    
                    # Look for regular intervals
                    common_intervals = time_diffs.mode()
                    if len(common_intervals) > 0:
                        most_common_interval = common_intervals.iloc[0]
                        interval_count = (time_diffs == most_common_interval).sum()
                        
                        if interval_count > len(time_diffs) * 0.3:  # 30% of intervals are the same
                            patterns.append(BehavioralPattern(
                                pattern_type="Regular Timing Pattern",
                                description=f"Regular interval detected: {most_common_interval} ({interval_count} occurrences)",
                                confidence=0.8,
                                severity="medium",
                                entities=[],
                                metrics={'interval': str(most_common_interval), 'count': interval_count}
                            ))
                
                # Detect burst activity
                hourly_counts = timestamps.dt.floor('H').value_counts()
                if len(hourly_counts) > 0:
                    max_hourly = hourly_counts.max()
                    avg_hourly = hourly_counts.mean()
                    
                    if max_hourly > avg_hourly * 5:  # 5x average activity in one hour
                        burst_hour = hourly_counts.idxmax()
                        patterns.append(BehavioralPattern(
                            pattern_type="Activity Burst",
                            description=f"Activity burst detected at {burst_hour}: {max_hourly} events vs avg {avg_hourly:.1f}",
                            confidence=0.7,
                            severity="medium",
                            entities=[],
                            metrics={'burst_count': max_hourly, 'average_count': avg_hourly, 'burst_time': str(burst_hour)}
                        ))
            
            except Exception as e:
                self.logger.warning(f"Error analyzing timing patterns: {e}")
        
        return patterns

class FinancialBehaviorAnalyzer:
    """Analyzes financial behavior patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_financial_behavior(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Analyze financial behavior patterns"""
        patterns = []
        
        patterns.extend(self._detect_spending_patterns(data))
        patterns.extend(self._detect_transaction_timing(data))
        patterns.extend(self._detect_amount_patterns(data))
        patterns.extend(self._detect_merchant_patterns(data))
        
        return patterns
    
    def _detect_spending_patterns(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect unusual spending patterns"""
        patterns = []
        
        # Find amount and account columns
        amount_cols = [col for col in data.columns if 'amount' in col.lower()]
        account_cols = [col for col in data.columns if 'account' in col.lower()]
        
        if amount_cols and account_cols:
            amount_col = amount_cols[0]
            account_col = account_cols[0]
            
            # Analyze spending by account
            account_spending = data.groupby(account_col)[amount_col].agg(['sum', 'mean', 'count', 'std'])
            
            # Detect high-volume accounts
            spending_threshold = account_spending['sum'].quantile(0.95)
            high_spenders = account_spending[account_spending['sum'] > spending_threshold]
            
            for account, metrics in high_spenders.iterrows():
                patterns.append(BehavioralPattern(
                    pattern_type="High Volume Spending",
                    description=f"Account {account} has high total spending: ${metrics['sum']:,.2f}",
                    confidence=0.8,
                    severity="medium",
                    entities=[account],
                    metrics={'total_spending': metrics['sum'], 'transaction_count': metrics['count']}
                ))
            
            # Detect accounts with high transaction frequency
            freq_threshold = account_spending['count'].quantile(0.95)
            frequent_accounts = account_spending[account_spending['count'] > freq_threshold]
            
            for account, metrics in frequent_accounts.iterrows():
                patterns.append(BehavioralPattern(
                    pattern_type="High Transaction Frequency",
                    description=f"Account {account} has high transaction frequency: {metrics['count']} transactions",
                    confidence=0.7,
                    severity="medium",
                    entities=[account],
                    metrics={'transaction_count': metrics['count'], 'avg_amount': metrics['mean']}
                ))
        
        return patterns
    
    def _detect_transaction_timing(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect unusual transaction timing patterns"""
        patterns = []
        
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if time_cols:
            time_col = time_cols[0]
            try:
                timestamps = pd.to_datetime(data[time_col])
                
                # Detect off-hours transactions
                night_mask = (timestamps.dt.hour < 6) | (timestamps.dt.hour > 22)
                night_transactions = night_mask.sum()
                total_transactions = len(data)
                
                if night_transactions / total_transactions > 0.2:  # More than 20% night transactions
                    patterns.append(BehavioralPattern(
                        pattern_type="Off-Hours Transaction Pattern",
                        description=f"High proportion of night transactions: {night_transactions}/{total_transactions} ({night_transactions/total_transactions:.1%})",
                        confidence=0.7,
                        severity="medium",
                        entities=[],
                        metrics={'night_transaction_ratio': night_transactions / total_transactions}
                    ))
                
                # Detect weekend transaction patterns
                weekend_mask = timestamps.dt.dayofweek >= 5
                weekend_transactions = weekend_mask.sum()
                
                if weekend_transactions / total_transactions > 0.4:  # More than 40% weekend transactions
                    patterns.append(BehavioralPattern(
                        pattern_type="Weekend Transaction Pattern",
                        description=f"High proportion of weekend transactions: {weekend_transactions}/{total_transactions} ({weekend_transactions/total_transactions:.1%})",
                        confidence=0.6,
                        severity="low",
                        entities=[],
                        metrics={'weekend_transaction_ratio': weekend_transactions / total_transactions}
                    ))
            
            except Exception as e:
                self.logger.warning(f"Error analyzing transaction timing: {e}")
        
        return patterns
    
    def _detect_amount_patterns(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect unusual amount patterns"""
        patterns = []
        
        amount_cols = [col for col in data.columns if 'amount' in col.lower()]
        
        for amount_col in amount_cols:
            if data[amount_col].dtype in ['int64', 'float64']:
                amounts = data[amount_col].dropna()
                
                # Detect round number bias
                round_amounts = amounts[amounts % 100 == 0]
                round_ratio = len(round_amounts) / len(amounts)
                
                if round_ratio > 0.3:  # More than 30% round amounts
                    patterns.append(BehavioralPattern(
                        pattern_type="Round Amount Pattern",
                        description=f"High proportion of round amounts: {len(round_amounts)}/{len(amounts)} ({round_ratio:.1%})",
                        confidence=0.6,
                        severity="low",
                        entities=[],
                        metrics={'round_amount_ratio': round_ratio}
                    ))
                
                # Detect structuring (amounts just under reporting thresholds)
                structuring_amounts = amounts[(amounts >= 9000) & (amounts < 10000)]
                if len(structuring_amounts) > 5:  # Multiple transactions just under $10K
                    patterns.append(BehavioralPattern(
                        pattern_type="Potential Structuring",
                        description=f"Multiple transactions just under $10K threshold: {len(structuring_amounts)} transactions",
                        confidence=0.8,
                        severity="high",
                        entities=[],
                        metrics={'structuring_count': len(structuring_amounts)}
                    ))
                
                # Detect repeated exact amounts
                amount_counts = amounts.value_counts()
                repeated_amounts = amount_counts[amount_counts > 5]  # Same amount more than 5 times
                
                for amount, count in repeated_amounts.items():
                    patterns.append(BehavioralPattern(
                        pattern_type="Repeated Amount Pattern",
                        description=f"Amount ${amount:,.2f} appears {count} times",
                        confidence=0.7,
                        severity="medium",
                        entities=[],
                        metrics={'repeated_amount': amount, 'repetition_count': count}
                    ))
        
        return patterns
    
    def _detect_merchant_patterns(self, data: pd.DataFrame) -> List[BehavioralPattern]:
        """Detect unusual merchant patterns"""
        patterns = []
        
        # Look for merchant columns
        merchant_cols = [col for col in data.columns 
                        if any(keyword in col.lower() for keyword in ['merchant', 'vendor', 'payee', 'recipient'])]
        
        if merchant_cols:
            merchant_col = merchant_cols[0]
            merchant_counts = data[merchant_col].value_counts()
            
            # Detect merchants with unusually high transaction counts
            high_threshold = merchant_counts.quantile(0.95)
            high_volume_merchants = merchant_counts[merchant_counts > high_threshold]
            
            for merchant, count in high_volume_merchants.items():
                patterns.append(BehavioralPattern(
                    pattern_type="High Volume Merchant",
                    description=f"Merchant '{merchant}' has unusually high transaction volume: {count} transactions",
                    confidence=0.7,
                    severity="medium",
                    entities=[merchant],
                    metrics={'transaction_count': count}
                ))
        
        return patterns

class BehavioralAnalysisEngine:
    """Main engine for comprehensive behavioral analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.user_analyzer = UserBehaviorAnalyzer()
        self.network_analyzer = NetworkBehaviorAnalyzer()
        self.financial_analyzer = FinancialBehaviorAnalyzer()
    
    def analyze_all_behaviors(self, data: pd.DataFrame, analysis_type: str = "auto") -> List[BehavioralPattern]:
        """Perform comprehensive behavioral analysis"""
        all_patterns = []
        
        try:
            # Always run user behavior analysis
            user_patterns = self.user_analyzer.analyze_user_behavior(data)
            all_patterns.extend(user_patterns)
            self.logger.info(f"Found {len(user_patterns)} user behavior patterns")
            
            # Run specific analysis based on type or data characteristics
            if analysis_type == "network" or self._is_network_data(data):
                network_patterns = self.network_analyzer.analyze_network_behavior(data)
                all_patterns.extend(network_patterns)
                self.logger.info(f"Found {len(network_patterns)} network behavior patterns")
            
            if analysis_type == "financial" or self._is_financial_data(data):
                financial_patterns = self.financial_analyzer.analyze_financial_behavior(data)
                all_patterns.extend(financial_patterns)
                self.logger.info(f"Found {len(financial_patterns)} financial behavior patterns")
            
            # Sort patterns by severity and confidence
            all_patterns.sort(key=lambda x: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                x.confidence
            ), reverse=True)
            
            self.logger.info(f"Total behavioral patterns detected: {len(all_patterns)}")
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
        
        return all_patterns
    
    def _is_network_data(self, data: pd.DataFrame) -> bool:
        """Check if data appears to be network-related"""
        network_keywords = ['ip', 'port', 'protocol', 'bytes', 'packets']
        network_cols = [col for col in data.columns 
                       if any(keyword in col.lower() for keyword in network_keywords)]
        return len(network_cols) >= 2
    
    def _is_financial_data(self, data: pd.DataFrame) -> bool:
        """Check if data appears to be financial-related"""
        financial_keywords = ['amount', 'transaction', 'account', 'merchant', 'payment']
        financial_cols = [col for col in data.columns 
                         if any(keyword in col.lower() for keyword in financial_keywords)]
        return len(financial_cols) >= 2
    
    def generate_behavioral_report(self, patterns: List[BehavioralPattern]) -> str:
        """Generate a comprehensive behavioral analysis report"""
        if not patterns:
            return "No significant behavioral patterns detected."
        
        report_lines = []
        report_lines.append("BEHAVIORAL ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Summary statistics
        severity_counts = Counter(pattern.severity for pattern in patterns)
        pattern_type_counts = Counter(pattern.pattern_type for pattern in patterns)
        
        report_lines.append("SUMMARY:")
        report_lines.append(f"Total patterns detected: {len(patterns)}")
        report_lines.append(f"Severity breakdown: {dict(severity_counts)}")
        report_lines.append("")
        
        # Top pattern types
        report_lines.append("TOP PATTERN TYPES:")
        for pattern_type, count in pattern_type_counts.most_common(5):
            report_lines.append(f"- {pattern_type}: {count} occurrences")
        report_lines.append("")
        
        # Detailed findings by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_patterns = [p for p in patterns if p.severity == severity]
            if severity_patterns:
                report_lines.append(f"{severity.upper()} SEVERITY PATTERNS:")
                for pattern in severity_patterns[:5]:  # Top 5 per severity
                    report_lines.append(f"- {pattern.description}")
                    report_lines.append(f"  Confidence: {pattern.confidence:.2f}")
                    if pattern.entities:
                        report_lines.append(f"  Entities: {', '.join(pattern.entities[:3])}")
                    report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if severity_counts.get('critical', 0) > 0:
            report_lines.append("- IMMEDIATE: Investigate critical severity patterns")
        if severity_counts.get('high', 0) > 0:
            report_lines.append("- HIGH PRIORITY: Review high severity patterns within 24 hours")
        if severity_counts.get('medium', 0) > 0:
            report_lines.append("- MEDIUM PRIORITY: Analyze medium severity patterns within 48 hours")
        
        report_lines.append("- Implement continuous monitoring for detected pattern types")
        report_lines.append("- Consider updating security policies based on findings")
        
        return "\n".join(report_lines)

# Global instance for easy access
behavioral_engine = BehavioralAnalysisEngine()

