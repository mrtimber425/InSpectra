# analytics_engine.py
# Statistical analysis and data analytics functionality

import pandas as pd
import numpy as np
import duckdb
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')


class AnalyticsEngine:
    """Handles all statistical analysis and data analytics operations"""

    def __init__(self, duckdb_connection=None):
        self.connection = duckdb_connection
        if self.connection is None:
            self.connection = duckdb.connect()

    def register_dataframe(self, df: pd.DataFrame, table_name: str = 'main_table'):
        """Register pandas DataFrame with DuckDB"""
        try:
            self.connection.register(table_name, df)
            return True
        except Exception as e:
            print(f"Error registering DataFrame: {e}")
            return False

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns

            summary = {
                'dataset_overview': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numeric_columns': len(numeric_cols),
                    'categorical_columns': len(categorical_cols),
                    'datetime_columns': len(datetime_cols),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
                    'missing_values_total': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                },
                'numeric_summary': {},
                'categorical_summary': {},
                'datetime_summary': {}
            }

            # Numeric columns analysis
            if len(numeric_cols) > 0:
                numeric_stats = df[numeric_cols].describe()
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    summary['numeric_summary'][col] = {
                        'count': len(col_data),
                        'missing': df[col].isnull().sum(),
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std(),
                        'variance': col_data.var(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'range': col_data.max() - col_data.min(),
                        'q1': col_data.quantile(0.25),
                        'q3': col_data.quantile(0.75),
                        'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'coefficient_of_variation': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0,
                        'outliers_iqr': self._count_outliers_iqr(col_data),
                        'outliers_zscore': self._count_outliers_zscore(col_data)
                    }

            # Categorical columns analysis
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    col_data = df[col].dropna()
                    value_counts = df[col].value_counts()
                    summary['categorical_summary'][col] = {
                        'count': len(col_data),
                        'missing': df[col].isnull().sum(),
                        'unique_values': df[col].nunique(),
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'most_frequent_percentage': (value_counts.iloc[0] / len(df)) * 100 if len(
                            value_counts) > 0 else 0,
                        'entropy': self._calculate_entropy(df[col]),
                        'top_5_values': value_counts.head(5).to_dict()
                    }

            # Datetime columns analysis
            if len(datetime_cols) > 0:
                for col in datetime_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        summary['datetime_summary'][col] = {
                            'count': len(col_data),
                            'missing': df[col].isnull().sum(),
                            'min_date': col_data.min(),
                            'max_date': col_data.max(),
                            'date_range_days': (col_data.max() - col_data.min()).days,
                            'unique_dates': col_data.nunique()
                        }

            return summary

        except Exception as e:
            raise Exception(f"Error generating summary statistics: {str(e)}")

    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return len(series[(series < lower_bound) | (series > upper_bound)])

    def _count_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> int:
        """Count outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series))
        return len(series[z_scores > threshold])

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical variable"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def generate_correlation_analysis(self, df: pd.DataFrame, method: str = 'pearson',
                                      min_correlation: float = 0.1) -> Dict[str, Any]:
        """Generate correlation analysis for numeric variables"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                return {'error': 'Need at least 2 numeric columns for correlation analysis'}

            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = df[numeric_cols].corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = df[numeric_cols].corr(method='spearman')
            else:
                raise ValueError(f"Unsupported correlation method: {method}")

            # Find significant correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]

                    if abs(corr_val) >= min_correlation:
                        # Calculate p-value
                        if method == 'pearson':
                            _, p_value = pearsonr(df[col1].dropna(), df[col2].dropna())
                        else:
                            _, p_value = spearmanr(df[col1].dropna(), df[col2].dropna())

                        correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr_val,
                            'abs_correlation': abs(corr_val),
                            'p_value': p_value,
                            'significance': 'significant' if p_value < 0.05 else 'not significant'
                        })

            # Sort by absolute correlation
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

            result = {
                'correlation_matrix': corr_matrix.to_dict(),
                'method': method,
                'significant_correlations': correlations[:20],  # Top 20
                'summary': {
                    'total_pairs': len(correlations),
                    'strong_positive': len([c for c in correlations if c['correlation'] > 0.7]),
                    'strong_negative': len([c for c in correlations if c['correlation'] < -0.7]),
                    'moderate_positive': len([c for c in correlations if 0.3 < c['correlation'] <= 0.7]),
                    'moderate_negative': len([c for c in correlations if -0.7 <= c['correlation'] < -0.3]),
                    'weak': len([c for c in correlations if -0.3 <= c['correlation'] <= 0.3])
                }
            }

            return result

        except Exception as e:
            raise Exception(f"Error in correlation analysis: {str(e)}")

    def generate_group_analysis(self, df: pd.DataFrame, group_by_columns: List[str],
                                analyze_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate group-based analysis"""
        try:
            if not group_by_columns:
                return {'error': 'No grouping columns specified'}

            if analyze_columns is None:
                analyze_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            results = {}

            for group_col in group_by_columns:
                if group_col not in df.columns:
                    continue

                group_results = {
                    'group_counts': df[group_col].value_counts().to_dict(),
                    'group_percentages': (df[group_col].value_counts(normalize=True) * 100).to_dict(),
                    'numeric_analysis': {}
                }

                # Analyze numeric columns by groups
                for analyze_col in analyze_columns:
                    if analyze_col in df.columns and pd.api.types.is_numeric_dtype(df[analyze_col]):
                        grouped = df.groupby(group_col)[analyze_col]

                        group_stats = {
                            'mean_by_group': grouped.mean().to_dict(),
                            'median_by_group': grouped.median().to_dict(),
                            'std_by_group': grouped.std().to_dict(),
                            'count_by_group': grouped.count().to_dict(),
                            'sum_by_group': grouped.sum().to_dict()
                        }

                        # ANOVA test if more than 2 groups
                        unique_groups = df[group_col].nunique()
                        if unique_groups > 2:
                            groups = [df[df[group_col] == group][analyze_col].dropna()
                                      for group in df[group_col].unique() if not pd.isna(group)]
                            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                                f_stat, p_value = stats.f_oneway(*groups)
                                group_stats['anova'] = {
                                    'f_statistic': f_stat,
                                    'p_value': p_value,
                                    'significant_difference': p_value < 0.05
                                }

                        group_results['numeric_analysis'][analyze_col] = group_stats

                results[group_col] = group_results

            return results

        except Exception as e:
            raise Exception(f"Error in group analysis: {str(e)}")

    def generate_time_series_analysis(self, df: pd.DataFrame, date_column: str,
                                      value_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate time series analysis"""
        try:
            if date_column not in df.columns:
                return {'error': f'Date column {date_column} not found'}

            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                except:
                    return {'error': f'Cannot convert {date_column} to datetime'}

            # Sort by date
            df_sorted = df.sort_values(date_column)

            if value_columns is None:
                value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            results = {
                'date_range': {
                    'start_date': df_sorted[date_column].min(),
                    'end_date': df_sorted[date_column].max(),
                    'total_days': (df_sorted[date_column].max() - df_sorted[date_column].min()).days,
                    'total_records': len(df_sorted)
                },
                'temporal_patterns': {},
                'trend_analysis': {}
            }

            # Temporal patterns
            df_sorted['year'] = df_sorted[date_column].dt.year
            df_sorted['month'] = df_sorted[date_column].dt.month
            df_sorted['day_of_week'] = df_sorted[date_column].dt.dayofweek
            df_sorted['hour'] = df_sorted[date_column].dt.hour

            results['temporal_patterns'] = {
                'records_by_year': df_sorted['year'].value_counts().sort_index().to_dict(),
                'records_by_month': df_sorted['month'].value_counts().sort_index().to_dict(),
                'records_by_day_of_week': df_sorted['day_of_week'].value_counts().sort_index().to_dict(),
                'records_by_hour': df_sorted['hour'].value_counts().sort_index().to_dict()
            }

            # Trend analysis for numeric columns
            for col in value_columns:
                if col in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted[col]):
                    col_data = df_sorted[[date_column, col]].dropna()

                    if len(col_data) > 1:
                        # Calculate trend using linear regression
                        x_numeric = (col_data[date_column] - col_data[date_column].min()).dt.days
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, col_data[col])

                        # Monthly aggregation
                        monthly_data = col_data.set_index(date_column).resample('M')[col].agg(['mean', 'sum', 'count'])

                        results['trend_analysis'][col] = {
                            'trend_slope': slope,
                            'trend_r_squared': r_value ** 2,
                            'trend_p_value': p_value,
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                            'monthly_averages': monthly_data['mean'].to_dict(),
                            'monthly_totals': monthly_data['sum'].to_dict(),
                            'basic_stats': {
                                'mean': col_data[col].mean(),
                                'std': col_data[col].std(),
                                'min': col_data[col].min(),
                                'max': col_data[col].max()
                            }
                        }

            return results

        except Exception as e:
            raise Exception(f"Error in time series analysis: {str(e)}")

    def generate_distribution_analysis(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            results = {}

            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    col_data = df[col].dropna()

                    if len(col_data) == 0:
                        continue

                    # Normality tests
                    shapiro_stat, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))

                    # Distribution characteristics
                    results[col] = {
                        'normality_test': {
                            'shapiro_wilk_statistic': shapiro_stat,
                            'shapiro_wilk_p_value': shapiro_p,
                            'is_normal': shapiro_p > 0.05
                        },
                        'distribution_stats': {
                            'skewness': col_data.skew(),
                            'kurtosis': col_data.kurtosis(),
                            'jarque_bera_test': stats.jarque_bera(col_data),
                        },
                        'percentiles': {
                            f'p{p}': col_data.quantile(p / 100) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                        }
                    }

            return results

        except Exception as e:
            raise Exception(f"Error in distribution analysis: {str(e)}")

    def generate_association_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze associations between variables and a target variable"""
        try:
            if target_column not in df.columns:
                return {'error': f'Target column {target_column} not found'}

            results = {
                'target_info': {
                    'column': target_column,
                    'type': str(df[target_column].dtype),
                    'unique_values': df[target_column].nunique(),
                    'missing_values': df[target_column].isnull().sum()
                },
                'numeric_associations': {},
                'categorical_associations': {}
            }

            # Numeric variable associations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_column:
                    # Calculate correlation
                    correlation, p_value = pearsonr(df[col].dropna(), df[target_column].dropna())

                    results['numeric_associations'][col] = {
                        'pearson_correlation': correlation,
                        'p_value': p_value,
                        'significance': 'significant' if p_value < 0.05 else 'not_significant',
                        'strength': self._interpret_correlation_strength(abs(correlation))
                    }

            # Categorical variable associations
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # Target is numeric, use ANOVA
                for col in categorical_cols:
                    groups = [df[df[col] == group][target_column].dropna()
                              for group in df[col].unique() if not pd.isna(group)]

                    if len(groups) > 1 and all(len(g) > 0 for g in groups):
                        f_stat, p_value = stats.f_oneway(*groups)

                        results['categorical_associations'][col] = {
                            'test': 'ANOVA',
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
            else:
                # Target is categorical, use Chi-square
                for col in categorical_cols:
                    if col != target_column:
                        contingency_table = pd.crosstab(df[col], df[target_column])
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                        results['categorical_associations'][col] = {
                            'test': 'Chi-square',
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }

            return results

        except Exception as e:
            raise Exception(f"Error in association analysis: {str(e)}")

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.7:
            return 'strong'
        elif correlation >= 0.3:
            return 'moderate'
        elif correlation >= 0.1:
            return 'weak'
        else:
            return 'very_weak'

    def execute_custom_analysis(self, df: pd.DataFrame, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom analysis based on configuration"""
        try:
            analysis_type = analysis_config.get('type', 'summary')

            if analysis_type == 'summary':
                return self.generate_summary_statistics(df)
            elif analysis_type == 'correlation':
                method = analysis_config.get('method', 'pearson')
                min_corr = analysis_config.get('min_correlation', 0.1)
                return self.generate_correlation_analysis(df, method, min_corr)
            elif analysis_type == 'group':
                group_cols = analysis_config.get('group_columns', [])
                analyze_cols = analysis_config.get('analyze_columns', None)
                return self.generate_group_analysis(df, group_cols, analyze_cols)
            elif analysis_type == 'time_series':
                date_col = analysis_config.get('date_column')
                value_cols = analysis_config.get('value_columns', None)
                return self.generate_time_series_analysis(df, date_col, value_cols)
            elif analysis_type == 'distribution':
                columns = analysis_config.get('columns', None)
                return self.generate_distribution_analysis(df, columns)
            elif analysis_type == 'association':
                target_col = analysis_config.get('target_column')
                return self.generate_association_analysis(df, target_col)
            else:
                return {'error': f'Unknown analysis type: {analysis_type}'}

        except Exception as e:
            raise Exception(f"Error in custom analysis: {str(e)}")