# ml_engine.py
# Machine learning functionality

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, silhouette_score)
import warnings

warnings.filterwarnings('ignore')


class MLEngine:
    """Enhanced Machine learning engine with comprehensive error handling"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.results = {}
        self.feature_names = []

        # Define available models with safer parameters
        self.classification_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
            'SVM': SVC(random_state=42, probability=True, gamma='scale'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }

        self.regression_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            'Linear Regression': LinearRegression(),
            'SVM': SVR(gamma='scale'),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }

        self.clustering_models = {
            'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(n_clusters=3)
        }

    def _validate_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[bool, str]:
        """Validate data before processing"""
        try:
            if df is None or df.empty:
                return False, "DataFrame is empty or None"

            if len(df) < 10:
                return False, "Dataset too small (minimum 10 samples required)"

            if target_column and target_column not in df.columns:
                return False, f"Target column '{target_column}' not found in dataset"

            # Check for valid data types
            if df.select_dtypes(include=[np.number, 'object', 'bool']).empty:
                return False, "No valid columns found for analysis"

            return True, "Data validation passed"

        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Robust feature preprocessing with comprehensive error handling"""
        try:
            X_processed = X.copy()

            # Store original feature names
            self.feature_names = list(X_processed.columns)

            # Handle datetime columns
            datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                X_processed[f'{col}_year'] = X_processed[col].dt.year
                X_processed[f'{col}_month'] = X_processed[col].dt.month
                X_processed[f'{col}_day'] = X_processed[col].dt.day
                X_processed = X_processed.drop(columns=[col])

            # Handle categorical variables
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns

            for col in categorical_cols:
                try:
                    # Handle missing values in categorical columns
                    X_processed[col] = X_processed[col].fillna('missing')

                    # Check cardinality
                    unique_vals = X_processed[col].nunique()
                    if unique_vals > 50:  # High cardinality
                        # Keep only top 20 most frequent values
                        top_values = X_processed[col].value_counts().head(20).index
                        X_processed[col] = X_processed[col].where(
                            X_processed[col].isin(top_values), 'other')

                    # Label encode
                    if f'categorical_{col}' not in self.encoders:
                        self.encoders[f'categorical_{col}'] = LabelEncoder()
                        X_processed[col] = self.encoders[f'categorical_{col}'].fit_transform(
                            X_processed[col].astype(str))
                    else:
                        # Handle unseen categories
                        encoder = self.encoders[f'categorical_{col}']
                        known_categories = set(encoder.classes_)
                        X_processed[col] = X_processed[col].apply(
                            lambda x: x if x in known_categories else 'unknown')
                        X_processed[col] = encoder.transform(X_processed[col].astype(str))

                except Exception as e:
                    print(f"Warning: Error processing categorical column {col}: {e}")
                    # Remove problematic column
                    X_processed = X_processed.drop(columns=[col])

            # Select only numeric columns for further processing
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_numeric = X_processed[numeric_cols]

            if X_numeric.empty:
                raise ValueError("No numeric features available after preprocessing")

            # Handle infinite values
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

            # Handle missing values in numeric columns
            if 'numeric' not in self.imputers:
                self.imputers['numeric'] = SimpleImputer(strategy='median')
                X_imputed = self.imputers['numeric'].fit_transform(X_numeric)
            else:
                X_imputed = self.imputers['numeric'].transform(X_numeric)

            # Convert back to DataFrame to maintain column names
            X_imputed = pd.DataFrame(X_imputed, columns=numeric_cols, index=X_numeric.index)

            # Remove columns with zero variance
            variance = X_imputed.var()
            constant_cols = variance[variance == 0].index
            if len(constant_cols) > 0:
                print(f"Warning: Removing constant columns: {list(constant_cols)}")
                X_imputed = X_imputed.drop(columns=constant_cols)

            if X_imputed.empty:
                raise ValueError("No features remaining after preprocessing")

            # Scale features
            if 'features' not in self.scalers:
                self.scalers['features'] = StandardScaler()
                X_scaled = self.scalers['features'].fit_transform(X_imputed)
            else:
                X_scaled = self.scalers['features'].transform(X_imputed)

            # Final check for NaN or infinite values
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                # Replace any remaining NaN/inf with median
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            return X_scaled

        except Exception as e:
            raise Exception(f"Feature preprocessing failed: {str(e)}")

    def _preprocess_target(self, y: pd.Series, task_type: str = 'classification') -> np.ndarray:
        """Robust target preprocessing"""
        try:
            y_processed = y.copy()

            # Handle missing values in target
            if y_processed.isnull().any():
                if task_type == 'classification':
                    # Use mode for classification
                    mode_value = y_processed.mode()
                    if len(mode_value) > 0:
                        y_processed = y_processed.fillna(mode_value[0])
                    else:
                        raise ValueError("Cannot determine mode for target variable")
                else:
                    # Use median for regression
                    median_value = y_processed.median()
                    if not pd.isna(median_value):
                        y_processed = y_processed.fillna(median_value)
                    else:
                        raise ValueError("Cannot determine median for target variable")

            # Encode target if it's categorical (for classification)
            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_processed):
                if 'target' not in self.encoders:
                    self.encoders['target'] = LabelEncoder()
                    y_encoded = self.encoders['target'].fit_transform(y_processed.astype(str))
                else:
                    y_encoded = self.encoders['target'].transform(y_processed.astype(str))
                return y_encoded

            return y_processed.values

        except Exception as e:
            raise Exception(f"Target preprocessing failed: {str(e)}")

    def prepare_data(self, df: pd.DataFrame, target_column: str,
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced data preparation with comprehensive error handling"""
        try:
            # Validate data
            is_valid, message = self._validate_data(df, target_column)
            if not is_valid:
                raise ValueError(message)

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Determine task type
            task_type = 'regression' if pd.api.types.is_numeric_dtype(y) else 'classification'

            # Preprocess features and target
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_target(y, task_type)

            # Ensure we have enough samples
            if len(X_processed) < 10:
                raise ValueError("Not enough samples after preprocessing")

            # Split data with stratification for classification
            if task_type == 'classification':
                # Check if we have enough samples per class for stratification
                unique_classes, class_counts = np.unique(y_processed, return_counts=True)
                min_class_count = np.min(class_counts)

                if min_class_count >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, test_size=test_size,
                        random_state=42, stratify=y_processed
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, test_size=test_size, random_state=42
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise Exception(f"Data preparation failed: {str(e)}")

    def train_classification_model(self, df: pd.DataFrame, target_column: str,
                                   model_name: str = 'Random Forest',
                                   test_size: float = 0.2) -> Dict[str, Any]:
        """Enhanced classification training with error handling"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_column, test_size)

            # Get model
            if model_name not in self.classification_models:
                raise ValueError(f"Unknown classification model: {model_name}")

            model = self.classification_models[model_name]

            # Train model with error handling
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                raise Exception(f"Model training failed: {str(e)}")

            # Make predictions
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                raise Exception(f"Prediction failed: {str(e)}")

            # Calculate metrics with error handling
            try:
                accuracy = accuracy_score(y_test, y_pred)

                # Handle multiclass vs binary classification
                average_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'

                precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
                recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
            except Exception as e:
                # Fallback to basic accuracy only
                accuracy = accuracy_score(y_test, y_pred)
                precision = recall = f1 = 0.0
                print(f"Warning: Could not calculate all metrics: {e}")

            # Cross-validation with error handling
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                cv_mean = cv_std = 0.0
                print(f"Warning: Cross-validation failed: {e}")

            # Feature importance
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    if len(self.feature_names) == len(model.feature_importances_):
                        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                        feature_importance = dict(sorted(feature_importance.items(),
                                                         key=lambda x: x[1], reverse=True))
            except Exception as e:
                print(f"Warning: Could not extract feature importance: {e}")

            # Store model
            self.models[f"{model_name}_classification"] = model

            results = {
                'model_type': 'classification',
                'model_name': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'feature_importance': feature_importance,
                'test_size': test_size,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y_test))
            }

            try:
                results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
                results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            except:
                results['classification_report'] = "Could not generate classification report"
                results['confusion_matrix'] = []

            self.results[f"{model_name}_classification"] = results
            return results

        except Exception as e:
            raise Exception(f"Classification training failed: {str(e)}")

    def train_regression_model(self, df: pd.DataFrame, target_column: str,
                               model_name: str = 'Random Forest',
                               test_size: float = 0.2) -> Dict[str, Any]:
        """Enhanced regression training with error handling"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_column, test_size)

            # Get model
            if model_name not in self.regression_models:
                raise ValueError(f"Unknown regression model: {model_name}")

            model = self.regression_models[model_name]

            # Train model
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                raise Exception(f"Model training failed: {str(e)}")

            # Make predictions
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                raise Exception(f"Prediction failed: {str(e)}")

            # Calculate metrics
            try:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
            except Exception as e:
                raise Exception(f"Metric calculation failed: {str(e)}")

            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                            scoring='neg_mean_squared_error')
                cv_mean = -cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                cv_mean = cv_std = 0.0
                print(f"Warning: Cross-validation failed: {e}")

            # Feature importance
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    if len(self.feature_names) == len(model.feature_importances_):
                        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                        feature_importance = dict(sorted(feature_importance.items(),
                                                         key=lambda x: x[1], reverse=True))
            except Exception as e:
                print(f"Warning: Could not extract feature importance: {e}")

            # Store model
            self.models[f"{model_name}_regression"] = model

            results = {
                'model_type': 'regression',
                'model_name': model_name,
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'feature_importance': feature_importance,
                'test_size': test_size,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1]
            }

            self.results[f"{model_name}_regression"] = results
            return results

        except Exception as e:
            raise Exception(f"Regression training failed: {str(e)}")

    def perform_clustering(self, df: pd.DataFrame, features: List[str],
                           algorithm: str = 'K-Means', n_clusters: int = 3) -> Dict[str, Any]:
        """Enhanced clustering with error handling"""
        try:
            # Validate inputs
            if not features or not all(feat in df.columns for feat in features):
                raise ValueError("Invalid or missing features for clustering")

            # Prepare data
            X = df[features].copy()

            # Preprocess features
            X_processed = self._preprocess_features(X)

            if X_processed.shape[1] == 0:
                raise ValueError("No valid features for clustering after preprocessing")

            # Get clustering model
            if algorithm == 'K-Means':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algorithm == 'DBSCAN':
                model = DBSCAN(eps=0.5, min_samples=5)
            elif algorithm == 'Agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")

            # Fit model
            try:
                cluster_labels = model.fit_predict(X_processed)
            except Exception as e:
                raise Exception(f"Clustering failed: {str(e)}")

            # Calculate metrics
            try:
                unique_labels = np.unique(cluster_labels)
                n_clusters_found = len(unique_labels)

                if n_clusters_found > 1 and -1 not in unique_labels:
                    silhouette = silhouette_score(X_processed, cluster_labels)
                else:
                    silhouette = 0.0
                    if -1 in unique_labels:
                        print("Warning: DBSCAN found noise points (label -1)")
            except Exception as e:
                silhouette = 0.0
                n_clusters_found = len(np.unique(cluster_labels))
                print(f"Warning: Could not calculate silhouette score: {e}")

            # Analyze clusters
            df_clustered = df.copy()
            df_clustered['cluster'] = cluster_labels

            cluster_summary = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                cluster_name = f'cluster_{cluster_id}' if cluster_id != -1 else 'noise'

                cluster_summary[cluster_name] = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(df)) * 100
                }

                # Add mean values for numeric features
                numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
                if numeric_features:
                    try:
                        cluster_summary[cluster_name]['mean_values'] = cluster_data[numeric_features].mean().to_dict()
                    except:
                        cluster_summary[cluster_name]['mean_values'] = {}

            # Store model
            self.models[f"{algorithm}_clustering"] = model

            results = {
                'algorithm': algorithm,
                'n_clusters': n_clusters_found,
                'silhouette_score': float(silhouette),
                'cluster_labels': cluster_labels.tolist(),
                'cluster_summary': cluster_summary,
                'features_used': features,
                'n_samples': len(df),
                'n_features': len(features)
            }

            self.results[f"{algorithm}_clustering"] = results
            return results

        except Exception as e:
            raise Exception(f"Clustering failed: {str(e)}")

    def perform_pca(self, df: pd.DataFrame, features: List[str],
                    n_components: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced PCA with error handling"""
        try:
            # Validate inputs
            if not features or not all(feat in df.columns for feat in features):
                raise ValueError("Invalid or missing features for PCA")

            # Prepare data
            X = df[features].copy()

            # Only use numeric features for PCA
            numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
            if not numeric_features:
                raise ValueError("No numeric features available for PCA")

            X_numeric = X[numeric_features]

            # Handle missing values
            if X_numeric.isnull().any().any():
                imputer = SimpleImputer(strategy='median')
                X_numeric = pd.DataFrame(imputer.fit_transform(X_numeric),
                                         columns=numeric_features, index=X_numeric.index)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)

            # Determine number of components
            if n_components is None:
                n_components = min(len(numeric_features), len(df), 10)  # Limit to 10 components max
            else:
                n_components = min(n_components, len(numeric_features), len(df))

            if n_components < 1:
                raise ValueError("Cannot perform PCA: insufficient components")

            # Perform PCA
            try:
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
            except Exception as e:
                raise Exception(f"PCA computation failed: {str(e)}")

            # Calculate explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # Feature loadings
            try:
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                loadings_df = pd.DataFrame(
                    loadings,
                    columns=[f'PC{i + 1}' for i in range(n_components)],
                    index=numeric_features
                )
                feature_loadings = loadings_df.to_dict()
            except Exception as e:
                print(f"Warning: Could not calculate feature loadings: {e}")
                feature_loadings = {}

            # Store model and scaler
            self.models['pca'] = pca
            self.scalers['pca'] = scaler

            results = {
                'n_components': n_components,
                'n_features_original': len(numeric_features),
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'total_variance_explained': float(cumulative_variance[-1]),
                'principal_components': X_pca.tolist(),
                'feature_loadings': feature_loadings,
                'features_used': numeric_features,
                'n_samples': len(df)
            }

            self.results['pca'] = results
            return results

        except Exception as e:
            raise Exception(f"PCA failed: {str(e)}")

    def auto_ml_classification(self, df: pd.DataFrame, target_column: str,
                               test_size: float = 0.2) -> Dict[str, Any]:
        """Enhanced Auto ML for classification"""
        try:
            results = {}
            best_model = None
            best_score = 0

            # Try each model with error handling
            for model_name in self.classification_models.keys():
                try:
                    result = self.train_classification_model(df, target_column, model_name, test_size)
                    results[model_name] = result

                    if result['accuracy'] > best_score:
                        best_score = result['accuracy']
                        best_model = model_name

                except Exception as e:
                    results[model_name] = {'error': str(e)}
                    print(f"Warning: {model_name} failed: {e}")

            if not best_model:
                raise Exception("All models failed to train")

            # Hyperparameter tuning for best model (simplified)
            try:
                tuned_result = self.tune_hyperparameters(df, target_column, best_model, 'classification')
                if tuned_result and 'error' not in tuned_result:
                    results[f"{best_model}_tuned"] = tuned_result
            except Exception as e:
                results[f"{best_model}_tuned"] = {'error': str(e)}
                print(f"Warning: Hyperparameter tuning failed: {e}")

            auto_results = {
                'task_type': 'classification',
                'best_model': best_model,
                'best_score': best_score,
                'all_results': results,
                'model_comparison': {name: res.get('accuracy', 0) for name, res in results.items()
                                     if 'accuracy' in res},
                'successful_models': [name for name, res in results.items() if 'accuracy' in res],
                'failed_models': [name for name, res in results.items() if 'error' in res]
            }

            return auto_results

        except Exception as e:
            raise Exception(f"Auto ML classification failed: {str(e)}")

    def auto_ml_regression(self, df: pd.DataFrame, target_column: str,
                           test_size: float = 0.2) -> Dict[str, Any]:
        """Enhanced Auto ML for regression"""
        try:
            results = {}
            best_model = None
            best_score = float('inf')

            for model_name in self.regression_models.keys():
                try:
                    result = self.train_regression_model(df, target_column, model_name, test_size)
                    results[model_name] = result

                    if result['mse'] < best_score:
                        best_score = result['mse']
                        best_model = model_name

                except Exception as e:
                    results[model_name] = {'error': str(e)}
                    print(f"Warning: {model_name} failed: {e}")

            if not best_model:
                raise Exception("All models failed to train")

            # Hyperparameter tuning for best model (simplified)
            try:
                tuned_result = self.tune_hyperparameters(df, target_column, best_model, 'regression')
                if tuned_result and 'error' not in tuned_result:
                    results[f"{best_model}_tuned"] = tuned_result
            except Exception as e:
                results[f"{best_model}_tuned"] = {'error': str(e)}
                print(f"Warning: Hyperparameter tuning failed: {e}")

            auto_results = {
                'task_type': 'regression',
                'best_model': best_model,
                'best_score': best_score,
                'all_results': results,
                'model_comparison': {name: res.get('mse', float('inf')) for name, res in results.items()
                                     if 'mse' in res},
                'successful_models': [name for name, res in results.items() if 'mse' in res],
                'failed_models': [name for name, res in results.items() if 'error' in res]
            }

            return auto_results

        except Exception as e:
            raise Exception(f"Auto ML regression failed: {str(e)}")

    def tune_hyperparameters(self, df: pd.DataFrame, target_column: str,
                             model_name: str, task_type: str) -> Dict[str, Any]:
        """Simplified hyperparameter tuning with error handling"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)

            # Simplified parameter grids
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None]
                },
                'SVM': {
                    'C': [0.1, 1.0],
                    'kernel': ['linear', 'rbf']
                },
                'Logistic Regression': {
                    'C': [0.1, 1.0, 10.0]
                }
            }

            if model_name not in param_grids:
                return {'error': f'Hyperparameter tuning not available for {model_name}'}

            # Get base model
            if task_type == 'classification':
                base_model = self.classification_models[model_name]
                scoring = 'accuracy'
            else:
                base_model = self.regression_models[model_name]
                scoring = 'neg_mean_squared_error'

            # Perform grid search with reduced CV folds
            try:
                grid_search = GridSearchCV(
                    base_model,
                    param_grids[model_name],
                    cv=3,  # Reduced from 5 to 3 for speed
                    scoring=scoring,
                    n_jobs=1,  # Single job to avoid issues
                    error_score='raise'
                )

                grid_search.fit(X_train, y_train)

                # Get best model and evaluate
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                if task_type == 'classification':
                    test_score = accuracy_score(y_test, y_pred)
                    metrics = {'accuracy': test_score}
                else:
                    test_score = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {'mse': test_score, 'r2_score': r2}

                results = {
                    'best_parameters': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_metrics': metrics,
                    'model_name': f"{model_name}_tuned"
                }

                # Store tuned model
                self.models[f"{model_name}_{task_type}_tuned"] = best_model

                return results

            except Exception as e:
                return {'error': f'Grid search failed: {str(e)}'}

        except Exception as e:
            return {'error': f'Hyperparameter tuning failed: {str(e)}'}

    def predict_new_data(self, model_key: str, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data with error handling"""
        try:
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")

            model = self.models[model_key]

            # Preprocess new data using the same transformations
            X_processed = self._preprocess_features(new_data)

            # Make predictions
            predictions = model.predict(X_processed)

            # Decode target if it was encoded
            if 'target' in self.encoders:
                try:
                    predictions = self.encoders['target'].inverse_transform(predictions.astype(int))
                except:
                    print("Warning: Could not decode predictions")

            return predictions

        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            'total_models': len(self.models),
            'model_types': {},
            'best_models': {},
            'all_results': self.results
        }

        # Categorize models by type
        for result_key, result in self.results.items():
            if 'error' in result:
                continue

            model_type = result.get('model_type', 'unknown')
            if model_type not in summary['model_types']:
                summary['model_types'][model_type] = []
            summary['model_types'][model_type].append(result_key)

        # Find best models for each type
        for model_type in summary['model_types']:
            if model_type == 'classification':
                best_accuracy = 0
                best_model = None
                for model_key in summary['model_types'][model_type]:
                    if model_key in self.results:
                        accuracy = self.results[model_key].get('accuracy', 0)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_key
                if best_model:
                    summary['best_models']['classification'] = {
                        'model': best_model,
                        'accuracy': best_accuracy
                    }

            elif model_type == 'regression':
                best_mse = float('inf')
                best_model = None
                for model_key in summary['model_types'][model_type]:
                    if model_key in self.results:
                        mse = self.results[model_key].get('mse', float('inf'))
                        if mse < best_mse:
                            best_mse = mse
                            best_model = model_key
                if best_model:
                    summary['best_models']['regression'] = {
                        'model': best_model,
                        'mse': best_mse
                    }

        return summary