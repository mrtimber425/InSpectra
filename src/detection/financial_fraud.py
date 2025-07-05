"""
Financial fraud detection module for CyberForensics Data Detective.
Implements algorithms to detect various types of financial fraud and suspicious transactions.
Enhanced with ML ensemble for improved accuracy.
"""

import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.helpers import calculate_velocity, detect_anomalous_amounts
from utils.ml_ensemble import MLEnsemble, AdvancedFeatureEngineering
from utils.behavioral_analysis import behavioral_engine
from config import config


@dataclass
class FinancialAnomaly:
    """Represents a detected financial fraud anomaly."""
    timestamp: datetime
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    account_id: Optional[str]
    transaction_id: Optional[str]
    amount: Optional[float]
    description: str
    confidence: float
    risk_score: float
    raw_data: Dict[str, Any]


class FinancialFraudDetector:
    """Detector for various financial fraud patterns and suspicious transactions with ML enhancement."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.thresholds = config.get_detection_thresholds('financial_fraud')
        
        # Initialize ML ensemble
        self.ml_ensemble = MLEnsemble(confidence_threshold=0.75)
        self.feature_engineer = AdvancedFeatureEngineering()
        self.is_ml_trained = False
        
        # Detection methods
        self.detection_methods = {
            'unusual_amounts': self._detect_unusual_amounts,
            'velocity_fraud': self._detect_velocity_fraud,
            'time_based_anomalies': self._detect_time_based_anomalies,
            'account_takeover': self._detect_account_takeover,
            'money_laundering': self._detect_money_laundering_patterns,
            'card_fraud': self._detect_card_fraud,
            'round_amount_fraud': self._detect_round_amount_fraud,
            'geographic_anomalies': self._detect_geographic_anomalies,
            'merchant_fraud': self._detect_merchant_fraud,
            'structuring': self._detect_structuring
        }
    
    def analyze_financial_data(self, df: pd.DataFrame, detection_types: Optional[List[str]] = None) -> List[FinancialAnomaly]:
        """Analyze financial transaction data for fraud patterns with ML enhancement."""
        self.logger.info(f"Starting ML-enhanced financial fraud analysis on {len(df)} transactions")
        
        if detection_types is None:
            detection_types = list(self.detection_methods.keys())
        
        # Train ML ensemble if not already trained
        if not self.is_ml_trained and len(df) > 50:
            self.logger.info("Training ML ensemble on financial data...")
            try:
                # Create financial-specific features
                financial_features = self.feature_engineer.create_financial_features(df)
                enhanced_data = pd.concat([df, financial_features], axis=1)
                
                # Define anomaly indicators for financial data
                anomaly_indicators = ['amount', 'transaction_amount', 'value', 'risk_score']
                
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
        
        # Apply ML ensemble enhancement if trained
        if self.is_ml_trained and all_anomalies:
            try:
                self.logger.info("Applying ML ensemble enhancement...")
                enhanced_anomalies = self.enhance_financial_anomalies_with_ml(df, all_anomalies)
                all_anomalies = enhanced_anomalies
            except Exception as e:
                self.logger.error(f"ML enhancement error: {e}")
        
        # Calculate risk scores and sort
        all_anomalies = self._calculate_risk_scores(all_anomalies)
        all_anomalies.sort(key=lambda x: (x.risk_score, x.timestamp), reverse=True)
        
        self.logger.info(f"Total financial anomalies detected: {len(all_anomalies)}")
        return all_anomalies
    
    def enhance_financial_anomalies_with_ml(self, data: pd.DataFrame, anomalies: List[FinancialAnomaly]) -> List[FinancialAnomaly]:
        """
        Enhance detected financial anomalies using ML ensemble predictions
        """
        try:
            # Get ML predictions for the entire dataset
            ml_predictions, ml_confidences = self.ml_ensemble.predict_anomalies(data, return_probabilities=True)
            
            enhanced_anomalies = []
            
            for anomaly in anomalies:
                # Find corresponding data row
                row_idx = 0
                if hasattr(anomaly, 'raw_data') and isinstance(anomaly.raw_data, dict):
                    # Try to find matching row based on raw data
                    for idx, row in data.iterrows():
                        match_count = 0
                        for key, value in anomaly.raw_data.items():
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
                enhanced_anomaly = FinancialAnomaly(
                    timestamp=anomaly.timestamp,
                    anomaly_type=f"ML-Enhanced {anomaly.anomaly_type}",
                    severity=new_severity,
                    account_id=anomaly.account_id,
                    transaction_id=anomaly.transaction_id,
                    amount=anomaly.amount,
                    description=f"{anomaly.description} (ML Confidence: {ml_confidence:.2f})",
                    confidence=min(combined_confidence, 1.0),
                    risk_score=anomaly.risk_score,
                    raw_data=anomaly.raw_data
                )
                
                enhanced_anomalies.append(enhanced_anomaly)
            
            return enhanced_anomalies
            
        except Exception as e:
            self.logger.error(f"Error enhancing financial anomalies with ML: {e}")
            return anomalies
    
    def _detect_unusual_amounts(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect transactions with unusual amounts."""
        anomalies = []
        
        amount_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['amount', 'value', 'sum'])]
        
        for amount_col in amount_columns:
            if not pd.api.types.is_numeric_dtype(df[amount_col]):
                continue
            
            # Statistical outlier detection
            amounts = df[amount_col].dropna()
            if len(amounts) < 10:
                continue
            
            # Use IQR method for outlier detection
            Q1 = amounts.quantile(0.25)
            Q3 = amounts.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier thresholds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Also check against absolute thresholds
            abs_threshold = self.thresholds.get('amount_threshold', 10000.0)
            
            outliers = df[
                (df[amount_col] > upper_bound) | 
                (df[amount_col] > abs_threshold) |
                (df[amount_col] < lower_bound)
            ]
            
            for _, row in outliers.iterrows():
                amount = row[amount_col]
                timestamp = row.get('timestamp', datetime.now())
                account_id = row.get('account_id', row.get('user_id', 'Unknown'))
                transaction_id = row.get('transaction_id', row.get('id', 'Unknown'))
                
                # Determine severity
                if amount > abs_threshold * 5:
                    severity = 'critical'
                elif amount > abs_threshold * 2:
                    severity = 'high'
                elif amount > upper_bound * 2:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Calculate confidence based on deviation
                if amount > upper_bound:
                    confidence = min(0.9, (amount - upper_bound) / upper_bound)
                else:
                    confidence = min(0.9, abs(amount - lower_bound) / abs(lower_bound) if lower_bound != 0 else 0.5)
                
                anomaly = FinancialAnomaly(
                    timestamp=timestamp,
                    anomaly_type='unusual_amount',
                    severity=severity,
                    account_id=str(account_id),
                    transaction_id=str(transaction_id),
                    amount=float(amount),
                    description=f"Unusual transaction amount: ${amount:,.2f} (threshold: ${abs_threshold:,.2f})",
                    confidence=confidence,
                    risk_score=0.0,  # Will be calculated later
                    raw_data={
                        'amount': float(amount),
                        'q1': float(Q1),
                        'q3': float(Q3),
                        'upper_bound': float(upper_bound),
                        'threshold': float(abs_threshold),
                        'deviation_factor': float(amount / upper_bound) if upper_bound > 0 else 0
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_velocity_fraud(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect high-velocity transaction patterns."""
        anomalies = []
        
        if 'timestamp' not in df.columns:
            return anomalies
        
        account_col = self._find_account_column(df)
        if not account_col:
            return anomalies
        
        # Group by account and analyze transaction velocity
        for account_id, group in df.groupby(account_col):
            if len(group) < 3:
                continue
            
            # Sort by timestamp
            group_sorted = group.sort_values('timestamp')
            timestamps = group_sorted['timestamp'].tolist()
            
            # Calculate velocity in different time windows
            for window_minutes in [5, 15, 60]:
                velocity = calculate_velocity(timestamps, window_minutes)
                threshold = self.thresholds.get('velocity_threshold', 5)
                
                for timestamp, count in velocity.items():
                    if count > threshold:
                        # Calculate total amount in this window
                        window_start = timestamp - timedelta(minutes=window_minutes)
                        window_transactions = group_sorted[
                            (group_sorted['timestamp'] >= window_start) & 
                            (group_sorted['timestamp'] <= timestamp)
                        ]
                        
                        total_amount = 0
                        amount_col = self._find_amount_column(df)
                        if amount_col:
                            total_amount = window_transactions[amount_col].sum()
                        
                        severity = self._calculate_velocity_severity(count, window_minutes)
                        confidence = min(0.9, count / (threshold * 2))
                        
                        anomaly = FinancialAnomaly(
                            timestamp=timestamp,
                            anomaly_type='velocity_fraud',
                            severity=severity,
                            account_id=str(account_id),
                            transaction_id=None,
                            amount=float(total_amount),
                            description=f"High transaction velocity: {count} transactions in {window_minutes} minutes",
                            confidence=confidence,
                            risk_score=0.0,
                            raw_data={
                                'transaction_count': int(count),
                                'window_minutes': window_minutes,
                                'total_amount': float(total_amount),
                                'threshold': threshold,
                                'velocity_ratio': float(count / threshold)
                            }
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_time_based_anomalies(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect transactions at unusual times."""
        anomalies = []
        
        if 'timestamp' not in df.columns:
            return anomalies
        
        # Define unusual hours (configurable)
        unusual_hours = self.thresholds.get('unusual_time_hours', [0, 1, 2, 3, 4, 5])
        
        # Extract hour from timestamp
        df_time = df.copy()
        df_time['hour'] = df_time['timestamp'].dt.hour
        
        # Find transactions in unusual hours
        unusual_time_transactions = df_time[df_time['hour'].isin(unusual_hours)]
        
        for _, row in unusual_time_transactions.iterrows():
            timestamp = row['timestamp']
            hour = row['hour']
            account_id = row.get('account_id', row.get('user_id', 'Unknown'))
            transaction_id = row.get('transaction_id', row.get('id', 'Unknown'))
            
            amount_col = self._find_amount_column(df)
            amount = row.get(amount_col, 0) if amount_col else 0
            
            # Higher severity for very late/early hours and large amounts
            if hour in [2, 3, 4] and amount > 1000:
                severity = 'high'
            elif hour in unusual_hours and amount > 5000:
                severity = 'medium'
            else:
                severity = 'low'
            
            confidence = 0.6 if amount > 1000 else 0.4
            
            anomaly = FinancialAnomaly(
                timestamp=timestamp,
                anomaly_type='time_based_anomaly',
                severity=severity,
                account_id=str(account_id),
                transaction_id=str(transaction_id),
                amount=float(amount),
                description=f"Transaction at unusual time: {hour:02d}:xx hours",
                confidence=confidence,
                risk_score=0.0,
                raw_data={
                    'hour': int(hour),
                    'amount': float(amount),
                    'unusual_hours': unusual_hours
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_account_takeover(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect potential account takeover patterns."""
        anomalies = []
        
        account_col = self._find_account_column(df)
        if not account_col:
            return anomalies
        
        # Look for sudden changes in transaction patterns
        for account_id, group in df.groupby(account_col):
            if len(group) < 10:
                continue
            
            # Sort by timestamp
            group_sorted = group.sort_values('timestamp')
            
            # Analyze pattern changes
            anomalies.extend(self._detect_pattern_changes(group_sorted, account_id))
            
            # Analyze location changes (if available)
            if 'location' in group.columns or 'ip_address' in group.columns:
                anomalies.extend(self._detect_location_changes(group_sorted, account_id))
        
        return anomalies
    
    def _detect_money_laundering_patterns(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect money laundering patterns."""
        anomalies = []
        
        account_col = self._find_account_column(df)
        amount_col = self._find_amount_column(df)
        
        if not account_col or not amount_col:
            return anomalies
        
        # Pattern 1: Rapid movement of funds (layering)
        anomalies.extend(self._detect_rapid_fund_movement(df, account_col, amount_col))
        
        # Pattern 2: Structuring (amounts just below reporting thresholds)
        anomalies.extend(self._detect_structuring_patterns(df, account_col, amount_col))
        
        # Pattern 3: Round amount patterns
        anomalies.extend(self._detect_round_amounts(df, account_col, amount_col))
        
        return anomalies
    
    def _detect_card_fraud(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect credit/debit card fraud patterns."""
        anomalies = []
        
        # Look for card-specific patterns
        card_col = self._find_card_column(df)
        if not card_col:
            return anomalies
        
        # Pattern 1: Multiple failed transactions followed by success
        anomalies.extend(self._detect_card_testing(df, card_col))
        
        # Pattern 2: Unusual merchant categories
        if 'merchant_category' in df.columns:
            anomalies.extend(self._detect_unusual_merchant_patterns(df, card_col))
        
        return anomalies
    
    def _detect_round_amount_fraud(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect suspicious round amount transactions."""
        anomalies = []
        
        amount_col = self._find_amount_column(df)
        if not amount_col:
            return anomalies
        
        # Find round amounts (multiples of 100, 500, 1000)
        round_amounts = df[
            (df[amount_col] % 100 == 0) & 
            (df[amount_col] >= 1000)
        ]
        
        account_col = self._find_account_column(df)
        if account_col:
            # Count round amounts per account
            round_counts = round_amounts.groupby(account_col).size()
            suspicious_accounts = round_counts[round_counts > 5]  # More than 5 round amounts
            
            for account_id, count in suspicious_accounts.items():
                account_transactions = round_amounts[round_amounts[account_col] == account_id]
                
                for _, row in account_transactions.iterrows():
                    timestamp = row.get('timestamp', datetime.now())
                    amount = row[amount_col]
                    transaction_id = row.get('transaction_id', row.get('id', 'Unknown'))
                    
                    confidence = min(0.8, count / 10)
                    severity = 'medium' if count > 10 else 'low'
                    
                    anomaly = FinancialAnomaly(
                        timestamp=timestamp,
                        anomaly_type='round_amount_fraud',
                        severity=severity,
                        account_id=str(account_id),
                        transaction_id=str(transaction_id),
                        amount=float(amount),
                        description=f"Suspicious round amount: ${amount:,.2f} ({count} round amounts for this account)",
                        confidence=confidence,
                        risk_score=0.0,
                        raw_data={
                            'amount': float(amount),
                            'round_amount_count': int(count),
                            'is_multiple_of_100': bool(amount % 100 == 0),
                            'is_multiple_of_500': bool(amount % 500 == 0),
                            'is_multiple_of_1000': bool(amount % 1000 == 0)
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_geographic_anomalies(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect geographically impossible transactions."""
        anomalies = []
        
        # This would require location data and distance calculations
        # Placeholder for geographic analysis
        if 'location' in df.columns or 'country' in df.columns:
            # Implementation would analyze impossible travel times between locations
            pass
        
        return anomalies
    
    def _detect_merchant_fraud(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect merchant-related fraud patterns."""
        anomalies = []
        
        if 'merchant_id' not in df.columns:
            return anomalies
        
        # Analyze merchant transaction patterns
        merchant_stats = df.groupby('merchant_id').agg({
            'timestamp': ['count', 'min', 'max'],
            self._find_amount_column(df) or 'amount': ['sum', 'mean', 'std']
        }).reset_index()
        
        # Flatten column names
        merchant_stats.columns = ['merchant_id', 'transaction_count', 'first_transaction', 
                                'last_transaction', 'total_amount', 'avg_amount', 'std_amount']
        
        # Detect suspicious patterns
        suspicious_merchants = merchant_stats[
            (merchant_stats['transaction_count'] > 100) &  # High volume
            (merchant_stats['std_amount'] < merchant_stats['avg_amount'] * 0.1)  # Low variance
        ]
        
        for _, merchant_row in suspicious_merchants.iterrows():
            merchant_id = merchant_row['merchant_id']
            merchant_transactions = df[df['merchant_id'] == merchant_id]
            
            # Create anomaly for the merchant pattern
            anomaly = FinancialAnomaly(
                timestamp=merchant_row['first_transaction'],
                anomaly_type='merchant_fraud',
                severity='medium',
                account_id=None,
                transaction_id=None,
                amount=float(merchant_row['total_amount']),
                description=f"Suspicious merchant pattern: {merchant_row['transaction_count']} transactions with low variance",
                confidence=0.7,
                risk_score=0.0,
                raw_data={
                    'merchant_id': str(merchant_id),
                    'transaction_count': int(merchant_row['transaction_count']),
                    'total_amount': float(merchant_row['total_amount']),
                    'avg_amount': float(merchant_row['avg_amount']),
                    'std_amount': float(merchant_row['std_amount'])
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_structuring(self, df: pd.DataFrame) -> List[FinancialAnomaly]:
        """Detect structuring (breaking large amounts into smaller ones to avoid reporting)."""
        anomalies = []
        
        amount_col = self._find_amount_column(df)
        account_col = self._find_account_column(df)
        
        if not amount_col or not account_col:
            return anomalies
        
        # Look for amounts just below common reporting thresholds
        reporting_thresholds = [10000, 5000, 3000]  # Common thresholds
        
        for threshold in reporting_thresholds:
            # Find amounts just below threshold (within 10%)
            near_threshold = df[
                (df[amount_col] >= threshold * 0.9) & 
                (df[amount_col] < threshold)
            ]
            
            if near_threshold.empty:
                continue
            
            # Group by account and look for patterns
            for account_id, group in near_threshold.groupby(account_col):
                if len(group) < 3:  # Need multiple transactions
                    continue
                
                # Check if transactions are close in time
                if 'timestamp' in df.columns:
                    group_sorted = group.sort_values('timestamp')
                    time_diffs = group_sorted['timestamp'].diff().dt.total_seconds().dropna()
                    
                    # If multiple transactions within a short time period
                    if len(time_diffs) > 0 and time_diffs.mean() < 3600:  # Within 1 hour average
                        total_amount = group[amount_col].sum()
                        
                        for _, row in group.iterrows():
                            timestamp = row.get('timestamp', datetime.now())
                            amount = row[amount_col]
                            transaction_id = row.get('transaction_id', row.get('id', 'Unknown'))
                            
                            severity = 'high' if total_amount > threshold * 2 else 'medium'
                            confidence = 0.8
                            
                            anomaly = FinancialAnomaly(
                                timestamp=timestamp,
                                anomaly_type='structuring',
                                severity=severity,
                                account_id=str(account_id),
                                transaction_id=str(transaction_id),
                                amount=float(amount),
                                description=f"Potential structuring: ${amount:,.2f} (threshold: ${threshold:,.2f}, total: ${total_amount:,.2f})",
                                confidence=confidence,
                                risk_score=0.0,
                                raw_data={
                                    'amount': float(amount),
                                    'threshold': float(threshold),
                                    'total_amount': float(total_amount),
                                    'transaction_count': len(group),
                                    'avg_time_diff_seconds': float(time_diffs.mean()) if len(time_diffs) > 0 else 0
                                }
                            )
                            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_risk_scores(self, anomalies: List[FinancialAnomaly]) -> List[FinancialAnomaly]:
        """Calculate risk scores for anomalies."""
        severity_weights = {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4}
        type_weights = {
            'unusual_amount': 0.7,
            'velocity_fraud': 0.9,
            'account_takeover': 0.95,
            'money_laundering': 0.9,
            'structuring': 0.85,
            'card_fraud': 0.8,
            'time_based_anomaly': 0.5,
            'round_amount_fraud': 0.6,
            'merchant_fraud': 0.7
        }
        
        for anomaly in anomalies:
            severity_weight = severity_weights.get(anomaly.severity, 0.5)
            type_weight = type_weights.get(anomaly.anomaly_type, 0.5)
            
            # Base risk score
            risk_score = anomaly.confidence * severity_weight * type_weight
            
            # Adjust based on amount (if available)
            if anomaly.amount and anomaly.amount > 10000:
                risk_score *= 1.2
            elif anomaly.amount and anomaly.amount > 50000:
                risk_score *= 1.5
            
            anomaly.risk_score = min(1.0, risk_score)
        
        return anomalies
    
    # Helper methods
    def _find_account_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the account identifier column."""
        account_columns = ['account_id', 'user_id', 'customer_id', 'account', 'user']
        for col in account_columns:
            if col in df.columns:
                return col
        return None
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the transaction amount column."""
        amount_columns = ['amount', 'value', 'sum', 'total', 'transaction_amount']
        for col in amount_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None
    
    def _find_card_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the card identifier column."""
        card_columns = ['card_id', 'card_number', 'card', 'payment_method']
        for col in card_columns:
            if col in df.columns:
                return col
        return None
    
    def _calculate_velocity_severity(self, count: int, window_minutes: int) -> str:
        """Calculate severity based on velocity."""
        if window_minutes <= 5 and count > 10:
            return 'critical'
        elif window_minutes <= 15 and count > 15:
            return 'high'
        elif count > 20:
            return 'medium'
        else:
            return 'low'
    
    def _detect_pattern_changes(self, group: pd.DataFrame, account_id: str) -> List[FinancialAnomaly]:
        """Detect sudden changes in transaction patterns."""
        # Placeholder for pattern change detection
        return []
    
    def _detect_location_changes(self, group: pd.DataFrame, account_id: str) -> List[FinancialAnomaly]:
        """Detect impossible location changes."""
        # Placeholder for location change detection
        return []
    
    def _detect_rapid_fund_movement(self, df: pd.DataFrame, account_col: str, amount_col: str) -> List[FinancialAnomaly]:
        """Detect rapid movement of funds between accounts."""
        # Placeholder for fund movement detection
        return []
    
    def _detect_structuring_patterns(self, df: pd.DataFrame, account_col: str, amount_col: str) -> List[FinancialAnomaly]:
        """Detect structuring patterns."""
        # This is handled by the main _detect_structuring method
        return []
    
    def _detect_round_amounts(self, df: pd.DataFrame, account_col: str, amount_col: str) -> List[FinancialAnomaly]:
        """Detect suspicious round amounts."""
        # This is handled by the main _detect_round_amount_fraud method
        return []
    
    def _detect_card_testing(self, df: pd.DataFrame, card_col: str) -> List[FinancialAnomaly]:
        """Detect card testing patterns."""
        # Placeholder for card testing detection
        return []
    
    def _detect_unusual_merchant_patterns(self, df: pd.DataFrame, card_col: str) -> List[FinancialAnomaly]:
        """Detect unusual merchant patterns."""
        # Placeholder for merchant pattern detection
        return []
    
    def get_fraud_summary(self, anomalies: List[FinancialAnomaly]) -> Dict[str, Any]:
        """Generate summary statistics for detected fraud."""
        if not anomalies:
            return {'total': 0, 'by_type': {}, 'by_severity': {}, 'total_risk_amount': 0}
        
        by_type = Counter(a.anomaly_type for a in anomalies)
        by_severity = Counter(a.severity for a in anomalies)
        
        total_risk_amount = sum(a.amount for a in anomalies if a.amount)
        high_risk_count = len([a for a in anomalies if a.risk_score > 0.7])
        
        return {
            'total': len(anomalies),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'total_risk_amount': total_risk_amount,
            'high_risk_count': high_risk_count,
            'avg_risk_score': np.mean([a.risk_score for a in anomalies]),
            'time_range': {
                'start': min(a.timestamp for a in anomalies),
                'end': max(a.timestamp for a in anomalies)
            }
        }

