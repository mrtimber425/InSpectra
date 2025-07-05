"""
Advanced ML Ensemble Module for Forensic Analysis
Combines Random Forest, XGBoost, and Isolation Forest for high accuracy anomaly detection
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Graceful imports for optional ML libraries
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn not available. SMOTE functionality disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Using RandomForest only.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Feature importance analysis limited.")

class MLEnsemble:
    """
    Advanced ML Ensemble for Forensic Analysis
    Combines multiple algorithms for maximum accuracy
    """
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=10)
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Initialize models based on available libraries
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            # Use additional RandomForest as fallback
            self.models['xgboost_fallback'] = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=43,
                n_jobs=-1
            )
        
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_features(self, data, target_column=None):
        """
        Advanced feature engineering for forensic analysis
        """
        features = pd.DataFrame()
        
        # Numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_column:
                # Basic statistics
                features[f'{col}_value'] = data[col]
                features[f'{col}_log'] = np.log1p(np.abs(data[col]))
                features[f'{col}_sqrt'] = np.sqrt(np.abs(data[col]))
                
                # Statistical features
                rolling_mean = data[col].rolling(window=5, min_periods=1).mean()
                features[f'{col}_rolling_mean'] = rolling_mean
                features[f'{col}_deviation'] = np.abs(data[col] - rolling_mean)
                
                # Percentile features
                features[f'{col}_percentile'] = data[col].rank(pct=True)
                
                # Z-score features
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    features[f'{col}_zscore'] = (data[col] - mean_val) / std_val
        
        # Categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_column:
                # Frequency encoding
                freq_map = data[col].value_counts().to_dict()
                features[f'{col}_frequency'] = data[col].map(freq_map)
                
                # Label encoding for high cardinality
                if data[col].nunique() < 50:
                    le = LabelEncoder()
                    features[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
        
        # Time-based features (if datetime columns exist)
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            features[f'{col}_hour'] = data[col].dt.hour
            features[f'{col}_day'] = data[col].dt.day
            features[f'{col}_month'] = data[col].dt.month
            features[f'{col}_weekday'] = data[col].dt.weekday
            features[f'{col}_is_weekend'] = (data[col].dt.weekday >= 5).astype(int)
        
        # Interaction features (top combinations)
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to prevent explosion
                for col2 in numeric_cols[i+1:6]:
                    if col1 != target_column and col2 != target_column:
                        features[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
                        features[f'{col1}_{col2}_diff'] = data[col1] - data[col2]
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def create_synthetic_labels(self, data, anomaly_indicators):
        """
        Create labels based on anomaly indicators for supervised learning
        """
        labels = np.zeros(len(data))
        
        # Check for various anomaly patterns
        for indicator in anomaly_indicators:
            if indicator in data.columns:
                # High values indicate anomalies
                if data[indicator].dtype in ['float64', 'int64']:
                    threshold = data[indicator].quantile(0.95)
                    anomaly_mask = (data[indicator] > threshold).astype(int)
                    labels = (labels.astype(int) | anomaly_mask).astype(float)
                
                # String patterns
                elif data[indicator].dtype == 'object':
                    suspicious_patterns = ['error', 'fail', 'suspicious', 'anomaly', 'fraud']
                    for pattern in suspicious_patterns:
                        pattern_mask = data[indicator].str.contains(pattern, case=False, na=False).astype(int)
                        labels = (labels.astype(int) | pattern_mask).astype(float)
        
        return labels
    
    def train_ensemble(self, data, target_column=None, anomaly_indicators=None):
        """
        Train the ensemble model on the provided data
        """
        try:
            # Prepare features
            features = self.prepare_features(data, target_column)
            self.feature_names = features.columns.tolist()
            
            # Create or use provided labels
            if target_column and target_column in data.columns:
                labels = self.label_encoder.fit_transform(data[target_column])
            elif anomaly_indicators:
                labels = self.create_synthetic_labels(data, anomaly_indicators)
            else:
                # Use statistical outliers as labels
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    # Use isolation forest to create initial labels
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    labels = (iso_forest.fit_predict(numeric_data) == -1).astype(int)
                else:
                    # Default to no anomalies
                    labels = np.zeros(len(data))
            
            # Handle class imbalance
            if len(np.unique(labels)) > 1 and np.sum(labels) > 0:
                # Scale features
                features_scaled = self.scaler.fit_transform(features)
                
                # Feature selection
                if features_scaled.shape[1] > 10:
                    features_scaled = self.feature_selector.fit_transform(features_scaled, labels)
                    selected_features = self.feature_selector.get_support()
                    self.feature_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
                
                # Handle imbalanced data
                if np.sum(labels) < len(labels) * 0.4:  # If minority class < 40%
                    if SMOTE_AVAILABLE:
                        smote = SMOTE(random_state=42)
                        features_scaled, labels = smote.fit_resample(features_scaled, labels)
                    else:
                        # Simple oversampling fallback
                        minority_indices = np.where(labels == 1)[0]
                        if len(minority_indices) > 0:
                            # Duplicate minority samples
                            duplicate_count = min(len(minority_indices), len(labels) // 4)
                            duplicate_indices = np.random.choice(minority_indices, duplicate_count, replace=True)
                            features_scaled = np.vstack([features_scaled, features_scaled[duplicate_indices]])
                            labels = np.hstack([labels, labels[duplicate_indices]])
                
                # Train supervised models
                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                # Train Random Forest
                self.models['random_forest'].fit(X_train, y_train)
                
                # Train XGBoost or fallback
                if XGBOOST_AVAILABLE:
                    self.models['xgboost'].fit(X_train, y_train)
                    xgb_score = self.models['xgboost'].score(X_test, y_test)
                    print(f"XGBoost Accuracy: {xgb_score:.3f}")
                else:
                    self.models['xgboost_fallback'].fit(X_train, y_train)
                    fallback_score = self.models['xgboost_fallback'].score(X_test, y_test)
                    print(f"XGBoost Fallback (RandomForest) Accuracy: {fallback_score:.3f}")
                
                # Evaluate models
                rf_score = self.models['random_forest'].score(X_test, y_test)
                print(f"Random Forest Accuracy: {rf_score:.3f}")
            
            # Train Isolation Forest (unsupervised)
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                self.models['isolation_forest'].fit(numeric_features)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def predict_anomalies(self, data, return_probabilities=False):
        """
        Predict anomalies using the ensemble approach
        """
        if not self.is_trained:
            print("Model not trained. Using statistical methods.")
            return self.statistical_anomaly_detection(data)
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Ensure feature consistency
            for feature in self.feature_names:
                if feature not in features.columns:
                    features[feature] = 0
            
            features = features[self.feature_names]
            
            predictions = []
            confidences = []
            
            # Get predictions from each model
            if hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features)
                
                # Random Forest predictions
                if 'random_forest' in self.models:
                    rf_pred = self.models['random_forest'].predict(features_scaled)
                    rf_prob = self.models['random_forest'].predict_proba(features_scaled)[:, 1]
                    predictions.append(rf_pred)
                    confidences.append(rf_prob)
                
                # XGBoost predictions
                if XGBOOST_AVAILABLE and 'xgboost' in self.models:
                    xgb_pred = self.models['xgboost'].predict(features_scaled)
                    xgb_prob = self.models['xgboost'].predict_proba(features_scaled)[:, 1]
                    predictions.append(xgb_pred)
                    confidences.append(xgb_prob)
                elif 'xgboost_fallback' in self.models:
                    fallback_pred = self.models['xgboost_fallback'].predict(features_scaled)
                    fallback_prob = self.models['xgboost_fallback'].predict_proba(features_scaled)[:, 1]
                    predictions.append(fallback_pred)
                    confidences.append(fallback_prob)
            
            # Isolation Forest predictions
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty and 'isolation_forest' in self.models:
                iso_pred = (self.models['isolation_forest'].predict(numeric_features) == -1).astype(int)
                iso_scores = self.models['isolation_forest'].score_samples(numeric_features)
                iso_prob = 1 / (1 + np.exp(iso_scores))  # Convert to probability-like scores
                predictions.append(iso_pred)
                confidences.append(iso_prob)
            
            # Ensemble voting
            if predictions:
                # Majority voting for predictions
                ensemble_pred = np.array(predictions).mean(axis=0) > 0.5
                
                # Average confidence scores
                ensemble_conf = np.array(confidences).mean(axis=0)
                
                if return_probabilities:
                    return ensemble_pred.astype(int), ensemble_conf
                else:
                    return ensemble_pred.astype(int)
            else:
                return self.statistical_anomaly_detection(data)
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return self.statistical_anomaly_detection(data)
    
    def statistical_anomaly_detection(self, data):
        """
        Fallback statistical anomaly detection
        """
        anomalies = np.zeros(len(data))
        
        # Numeric outlier detection
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            anomalies |= outliers.astype(int)
        
        return anomalies.astype(int)
    
    def get_feature_importance(self):
        """
        Get feature importance from trained models
        """
        importance_dict = {}
        
        if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
            importance_dict['random_forest'] = dict(zip(
                self.feature_names, 
                self.models['random_forest'].feature_importances_
            ))
        
        if 'xgboost' in self.models and hasattr(self.models['xgboost'], 'feature_importances_'):
            importance_dict['xgboost'] = dict(zip(
                self.feature_names,
                self.models['xgboost'].feature_importances_
            ))
        
        return importance_dict
    
    def get_model_performance(self, data, labels):
        """
        Get detailed performance metrics
        """
        if not self.is_trained:
            return None
        
        try:
            features = self.prepare_features(data)
            features = features[self.feature_names]
            features_scaled = self.scaler.transform(features)
            
            performance = {}
            
            for model_name, model in self.models.items():
                if model_name != 'isolation_forest':
                    pred = model.predict(features_scaled)
                    prob = model.predict_proba(features_scaled)[:, 1]
                    
                    performance[model_name] = {
                        'accuracy': (pred == labels).mean(),
                        'auc_score': roc_auc_score(labels, prob) if len(np.unique(labels)) > 1 else 0.5
                    }
            
            return performance
            
        except Exception as e:
            print(f"Performance evaluation error: {str(e)}")
            return None

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering specifically for forensic analysis
    """
    
    @staticmethod
    def create_behavioral_features(data, user_column=None, time_column=None):
        """
        Create behavioral analysis features
        """
        features = pd.DataFrame(index=data.index)
        
        if user_column and user_column in data.columns:
            # User-based features
            user_counts = data[user_column].value_counts()
            features['user_frequency'] = data[user_column].map(user_counts)
            
            # User activity patterns
            for user in data[user_column].unique()[:100]:  # Limit to prevent memory issues
                user_mask = data[user_column] == user
                features[f'is_user_{hash(user) % 1000}'] = user_mask.astype(int)
        
        if time_column and time_column in data.columns:
            # Time-based behavioral features
            features['hour_of_day'] = pd.to_datetime(data[time_column]).dt.hour
            features['day_of_week'] = pd.to_datetime(data[time_column]).dt.dayofweek
            features['is_weekend'] = (pd.to_datetime(data[time_column]).dt.dayofweek >= 5).astype(int)
            features['is_night'] = ((pd.to_datetime(data[time_column]).dt.hour < 6) | 
                                   (pd.to_datetime(data[time_column]).dt.hour > 22)).astype(int)
        
        return features
    
    @staticmethod
    def create_network_features(data):
        """
        Create network-specific features for cyber forensics
        """
        features = pd.DataFrame(index=data.index)
        
        # IP-based features
        ip_columns = [col for col in data.columns if 'ip' in col.lower()]
        for col in ip_columns:
            if col in data.columns:
                # IP frequency
                ip_counts = data[col].value_counts()
                features[f'{col}_frequency'] = data[col].map(ip_counts)
                
                # Private IP detection
                features[f'{col}_is_private'] = data[col].astype(str).str.match(
                    r'^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)'
                ).astype(int)
        
        # Port-based features
        port_columns = [col for col in data.columns if 'port' in col.lower()]
        for col in port_columns:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                # Well-known ports
                features[f'{col}_is_wellknown'] = (data[col] <= 1024).astype(int)
                features[f'{col}_is_ephemeral'] = (data[col] >= 32768).astype(int)
        
        return features
    
    @staticmethod
    def create_financial_features(data):
        """
        Create financial-specific features for fraud detection
        """
        features = pd.DataFrame(index=data.index)
        
        # Amount-based features
        amount_columns = [col for col in data.columns if 'amount' in col.lower()]
        for col in amount_columns:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                # Round number detection
                features[f'{col}_is_round'] = (data[col] % 100 == 0).astype(int)
                
                # Structuring detection (amounts just under reporting thresholds)
                features[f'{col}_near_threshold'] = ((data[col] >= 9000) & (data[col] < 10000)).astype(int)
                
                # Velocity features (if multiple transactions)
                features[f'{col}_rolling_sum'] = data[col].rolling(window=5, min_periods=1).sum()
                features[f'{col}_rolling_count'] = data[col].rolling(window=5, min_periods=1).count()
        
        # Account-based features
        account_columns = [col for col in data.columns if 'account' in col.lower()]
        for col in account_columns:
            if col in data.columns:
                account_counts = data[col].value_counts()
                features[f'{col}_transaction_count'] = data[col].map(account_counts)
        
        return features

# Global instance for easy access
ml_ensemble = MLEnsemble()

