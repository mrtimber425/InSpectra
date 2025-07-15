# export_manager.py
# Data export and report generation functionality

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import io
import base64
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ExportManager:
    """Handles data export and report generation"""

    def __init__(self):
        self.supported_formats = {
            'csv': self.export_csv,
            'excel': self.export_excel,
            'json': self.export_json,
            'parquet': self.export_parquet,
            'html': self.export_html,
            'xml': self.export_xml
        }

    def export_csv(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to CSV"""
        try:
            index = kwargs.get('index', False)
            encoding = kwargs.get('encoding', 'utf-8')
            separator = kwargs.get('separator', ',')

            df.to_csv(filepath, index=index, encoding=encoding, sep=separator)
            return True
        except Exception as e:
            raise Exception(f"Error exporting to CSV: {str(e)}")

    def export_excel(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to Excel"""
        try:
            sheet_name = kwargs.get('sheet_name', 'Sheet1')
            index = kwargs.get('index', False)

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)
            return True
        except Exception as e:
            raise Exception(f"Error exporting to Excel: {str(e)}")

    def export_json(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to JSON"""
        try:
            orient = kwargs.get('orient', 'records')
            indent = kwargs.get('indent', 2)

            df.to_json(filepath, orient=orient, indent=indent)
            return True
        except Exception as e:
            raise Exception(f"Error exporting to JSON: {str(e)}")

    def export_parquet(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to Parquet"""
        try:
            index = kwargs.get('index', False)
            df.to_parquet(filepath, index=index)
            return True
        except Exception as e:
            raise Exception(f"Error exporting to Parquet: {str(e)}")

    def export_html(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to HTML"""
        try:
            index = kwargs.get('index', False)
            table_id = kwargs.get('table_id', 'data_table')

            html_string = df.to_html(index=index, table_id=table_id, classes='table table-striped')

            # Create complete HTML document
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Export</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .table {{ border-collapse: collapse; width: 100%; }}
                    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .table th {{ background-color: #f2f2f2; font-weight: bold; }}
                    .table-striped tbody tr:nth-child(odd) {{ background-color: #f9f9f9; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>Data Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>
                <p>Total rows: {len(df)}, Total columns: {len(df.columns)}</p>
                {html_string}
            </body>
            </html>
            """

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_html)

            return True
        except Exception as e:
            raise Exception(f"Error exporting to HTML: {str(e)}")

    def export_xml(self, df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """Export DataFrame to XML"""
        try:
            root_name = kwargs.get('root_name', 'data')
            row_name = kwargs.get('row_name', 'record')

            df.to_xml(filepath, root_name=root_name, row_name=row_name)
            return True
        except Exception as e:
            raise Exception(f"Error exporting to XML: {str(e)}")

    def export_data(self, df: pd.DataFrame, filepath: str, format_type: str, **kwargs) -> bool:
        """Main export function"""
        try:
            if format_type.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported export format: {format_type}")

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Call appropriate export function
            return self.supported_formats[format_type.lower()](df, filepath, **kwargs)

        except Exception as e:
            raise Exception(f"Error exporting data: {str(e)}")

    def export_filtered_data(self, df: pd.DataFrame, filepath: str, format_type: str,
                             filters: Dict[str, Any], **kwargs) -> bool:
        """Export filtered subset of data"""
        try:
            filtered_df = self.apply_filters(df, filters)
            return self.export_data(filtered_df, filepath, format_type, **kwargs)
        except Exception as e:
            raise Exception(f"Error exporting filtered data: {str(e)}")

    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        filtered_df = df.copy()

        for column, filter_config in filters.items():
            if column not in df.columns:
                continue

            filter_type = filter_config.get('type', 'equals')
            value = filter_config.get('value')

            if filter_type == 'equals':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif filter_type == 'not_equals':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif filter_type == 'greater_than':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif filter_type == 'less_than':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif filter_type == 'contains':
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
            elif filter_type == 'in_list':
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif filter_type == 'between':
                min_val, max_val = value
                filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]

        return filtered_df

    def generate_summary_report(self, df: pd.DataFrame, data_info: Dict[str, Any],
                                analysis_results: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive summary report"""
        try:
            report = []

            # Header
            report.append("=" * 80)
            report.append("INSPECTRA ANALYTICS - COMPREHENSIVE DATA REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Data Source: {data_info.get('filename', 'Unknown')}")
            report.append("")

            # Executive Summary
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 20)
            report.append(f"• Dataset contains {len(df):,} rows and {len(df.columns)} columns")
            report.append(f"• Memory usage: {data_info.get('memory', 'N/A')}")
            report.append(f"• Data completeness: {((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%")

            # Identify data quality issues
            issues = []
            if df.isnull().sum().sum() > 0:
                issues.append(f"{df.isnull().sum().sum():,} missing values")
            if df.duplicated().sum() > 0:
                issues.append(f"{df.duplicated().sum():,} duplicate rows")

            if issues:
                report.append(f"• Data quality issues: {', '.join(issues)}")
            else:
                report.append("• No major data quality issues detected")

            report.append("")

            # Dataset Overview
            report.append("DATASET OVERVIEW")
            report.append("-" * 20)
            report.append(f"Total Rows: {len(df):,}")
            report.append(f"Total Columns: {len(df.columns)}")
            report.append(f"File Size: {data_info.get('file_size', 'N/A')}")
            report.append(f"Load Time: {data_info.get('load_time', 'N/A')}")
            report.append("")

            # Column Analysis
            report.append("COLUMN ANALYSIS")
            report.append("-" * 15)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns

            report.append(f"Numeric Columns: {len(numeric_cols)}")
            report.append(f"Categorical Columns: {len(categorical_cols)}")
            report.append(f"DateTime Columns: {len(datetime_cols)}")
            report.append("")

            # Detailed Column Information
            report.append("DETAILED COLUMN INFORMATION")
            report.append("-" * 30)

            for col in df.columns:
                report.append(f"\n{col}:")
                report.append(f"  Data Type: {df[col].dtype}")
                report.append(
                    f"  Missing Values: {df[col].isnull().sum():,} ({(df[col].isnull().sum() / len(df) * 100):.1f}%)")
                report.append(f"  Unique Values: {df[col].nunique():,}")

                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = df[col].describe()
                    report.append(f"  Mean: {stats['mean']:.2f}")
                    report.append(f"  Std Dev: {stats['std']:.2f}")
                    report.append(f"  Min: {stats['min']:.2f}")
                    report.append(f"  Max: {stats['max']:.2f}")

                    # Outlier detection
                    Q1 = stats['25%']
                    Q3 = stats['75%']
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    report.append(f"  Outliers (IQR): {len(outliers)} ({(len(outliers) / len(df) * 100):.1f}%)")

                elif df[col].dtype == 'object':
                    top_values = df[col].value_counts().head(3)
                    report.append(f"  Most Common: {top_values.index[0]} ({top_values.iloc[0]} occurrences)")
                    if len(top_values) > 1:
                        report.append(f"  Second Most: {top_values.index[1]} ({top_values.iloc[1]} occurrences)")

            # Analysis Results (if provided)
            if analysis_results:
                report.append("\n\nSTATISTICAL ANALYSIS RESULTS")
                report.append("-" * 30)

                for analysis_type, results in analysis_results.items():
                    report.append(f"\n{analysis_type.upper().replace('_', ' ')}:")
                    if isinstance(results, dict):
                        for key, value in results.items():
                            if isinstance(value, (int, float)):
                                report.append(f"  {key}: {value:.3f}")
                            elif isinstance(value, str):
                                report.append(f"  {key}: {value}")

            # Data Quality Assessment
            report.append("\n\nDATA QUALITY ASSESSMENT")
            report.append("-" * 25)

            # Missing data analysis
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                report.append("Columns with Missing Data:")
                for col in missing_cols:
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    report.append(f"  {col}: {missing_count:,} ({missing_pct:.1f}%)")
            else:
                report.append("No missing data detected")

            # Duplicate analysis
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                report.append(f"\nDuplicate Rows: {duplicate_count:,} ({(duplicate_count / len(df) * 100):.1f}%)")
            else:
                report.append("\nNo duplicate rows detected")

            # Recommendations
            report.append("\n\nRECOMMENDations")
            report.append("-" * 15)

            recommendations = []

            # Missing data recommendations
            high_missing_cols = [col for col in df.columns if (df[col].isnull().sum() / len(df)) > 0.5]
            if high_missing_cols:
                recommendations.append(
                    f"Consider dropping columns with >50% missing data: {', '.join(high_missing_cols)}")

            moderate_missing_cols = [col for col in df.columns if 0.1 < (df[col].isnull().sum() / len(df)) <= 0.5]
            if moderate_missing_cols:
                recommendations.append(
                    f"Consider imputation for columns with moderate missing data: {', '.join(moderate_missing_cols)}")

            # High cardinality categorical variables
            high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.1]
            if high_cardinality_cols:
                recommendations.append(
                    f"High cardinality categorical columns may need encoding: {', '.join(high_cardinality_cols)}")

            # Potential outliers
            numeric_outlier_cols = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    numeric_outlier_cols.append(col)

            if numeric_outlier_cols:
                recommendations.append(f"Review outliers in: {', '.join(numeric_outlier_cols)}")

            if not recommendations:
                recommendations.append("Data quality appears good. No immediate actions required.")

            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")

            # Footer
            report.append("\n" + "=" * 80)
            report.append("End of Report")
            report.append("=" * 80)

            return "\n".join(report)

        except Exception as e:
            raise Exception(f"Error generating summary report: {str(e)}")

    def generate_analysis_report(self, analysis_results: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate detailed analysis report"""
        try:
            report = []

            # Header
            report.append("=" * 80)
            report.append("INSPECTRA ANALYTICS - DETAILED ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Summary
            report.append("ANALYSIS SUMMARY")
            report.append("-" * 17)
            report.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
            report.append(f"Analyses Performed: {len(analysis_results)}")
            report.append("")

            # Detailed Results
            for analysis_name, results in analysis_results.items():
                report.append(f"{analysis_name.upper().replace('_', ' ')}")
                report.append("-" * len(analysis_name))

                if isinstance(results, dict):
                    self._format_analysis_section(report, results, level=0)
                else:
                    report.append(str(results))

                report.append("")

            return "\n".join(report)

        except Exception as e:
            raise Exception(f"Error generating analysis report: {str(e)}")

    def _format_analysis_section(self, report: List[str], data: Any, level: int = 0):
        """Recursively format analysis results"""
        indent = "  " * level

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    report.append(f"{indent}{key}:")
                    self._format_analysis_section(report, value, level + 1)
                elif isinstance(value, list):
                    report.append(f"{indent}{key}: {len(value)} items")
                    if len(value) <= 10:  # Only show first 10 items
                        for item in value:
                            report.append(f"{indent}  - {item}")
                elif isinstance(value, (int, float)):
                    if isinstance(value, float):
                        report.append(f"{indent}{key}: {value:.4f}")
                    else:
                        report.append(f"{indent}{key}: {value}")
                else:
                    report.append(f"{indent}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                report.append(f"{indent}[{i}]: {item}")
        else:
            report.append(f"{indent}{data}")

    def export_chart(self, figure: Figure, filepath: str, format_type: str = 'png', **kwargs) -> bool:
        """Export matplotlib figure to file"""
        try:
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            figure.savefig(filepath, format=format_type, dpi=dpi, bbox_inches=bbox_inches)
            return True

        except Exception as e:
            raise Exception(f"Error exporting chart: {str(e)}")

    def create_dashboard_html(self, df: pd.DataFrame, charts: List[Figure],
                              analysis_results: Dict[str, Any]) -> str:
        """Create interactive HTML dashboard"""
        try:
            # Convert charts to base64 encoded images
            chart_images = []
            for i, fig in enumerate(charts):
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                chart_images.append(f"data:image/png;base64,{img_base64}")
                img_buffer.close()

            # Generate data preview table
            data_preview = df.head(10).to_html(classes='table table-striped', table_id='preview-table')

            # Generate summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary_stats = df[numeric_cols].describe().to_html(classes='table table-bordered') if len(
                numeric_cols) > 0 else "<p>No numeric columns found</p>"

            # Create HTML dashboard
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>InSpectra Analytics Dashboard</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                    .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }}
                    .chart-container {{ margin: 2rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }}
                    .stats-card {{ background: #f8f9fa; padding: 1rem; margin: 1rem 0; border-radius: 8px; }}
                    .metric {{ text-align: center; margin: 1rem 0; }}
                    .metric-value {{ font-size: 2rem; font-weight: bold; color: #667eea; }}
                    .metric-label {{ font-size: 0.9rem; color: #6c757d; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <div class="dashboard-header">
                    <div class="container">
                        <h1 class="display-4">📊 InSpectra Analytics Dashboard</h1>
                        <p class="lead">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    </div>
                </div>

                <div class="container mt-4">
                    <!-- Key Metrics -->
                    <div class="row">
                        <div class="col-md-3">
                            <div class="stats-card text-center">
                                <div class="metric-value">{len(df):,}</div>
                                <div class="metric-label">Total Rows</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card text-center">
                                <div class="metric-value">{len(df.columns)}</div>
                                <div class="metric-label">Total Columns</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card text-center">
                                <div class="metric-value">{df.isnull().sum().sum():,}</div>
                                <div class="metric-label">Missing Values</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card text-center">
                                <div class="metric-value">{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%</div>
                                <div class="metric-label">Data Completeness</div>
                            </div>
                        </div>
                    </div>

                    <!-- Charts Section -->
                    <h2 class="mt-5">📈 Visualizations</h2>
                    <div class="row">
            """

            # Add charts
            for i, chart_img in enumerate(chart_images):
                html_content += f"""
                        <div class="col-lg-6">
                            <div class="chart-container">
                                <h5>Chart {i + 1}</h5>
                                <img src="{chart_img}" alt="Chart {i + 1}" class="img-fluid">
                            </div>
                        </div>
                """

            html_content += """
                    </div>

                    <!-- Data Preview -->
                    <h2 class="mt-5">🔍 Data Preview</h2>
                    <div class="table-responsive">
            """ + data_preview + """
                    </div>

                    <!-- Summary Statistics -->
                    <h2 class="mt-5">📊 Summary Statistics</h2>
                    <div class="table-responsive">
            """ + summary_stats + """
                    </div>

                    <!-- Analysis Results -->
                    <h2 class="mt-5">🔬 Analysis Results</h2>
                    <div class="row">
            """

            # Add analysis results
            for analysis_name, results in analysis_results.items():
                html_content += f"""
                        <div class="col-md-6">
                            <div class="stats-card">
                                <h5>{analysis_name.replace('_', ' ').title()}</h5>
                """

                if isinstance(results, dict):
                    for key, value in list(results.items())[:5]:  # Show first 5 items
                        if isinstance(value, (int, float)):
                            html_content += f"<p><strong>{key}:</strong> {value:.3f if isinstance(value, float) else value}</p>"
                        elif isinstance(value, str):
                            html_content += f"<p><strong>{key}:</strong> {value}</p>"

                html_content += """
                            </div>
                        </div>
                """

            html_content += """
                    </div>
                </div>

                <footer class="bg-light mt-5 py-4">
                    <div class="container text-center">
                        <p>&copy; 2024 InSpectra Analytics Platform. Generated automatically.</p>
                    </div>
                </footer>

                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
            """

            return html_content

        except Exception as e:
            raise Exception(f"Error creating dashboard HTML: {str(e)}")

    def export_dashboard(self, df: pd.DataFrame, charts: List[Figure],
                         analysis_results: Dict[str, Any], filepath: str) -> bool:
        """Export complete dashboard to HTML file"""
        try:
            dashboard_html = self.create_dashboard_html(df, charts, analysis_results)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)

            return True

        except Exception as e:
            raise Exception(f"Error exporting dashboard: {str(e)}")

    def get_export_summary(self, export_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of export operations"""
        if not export_history:
            return {
                'total_exports': 0,
                'formats_used': {},
                'total_size': 0,
                'recent_exports': []
            }

        formats_used = {}
        total_size = 0

        for export in export_history:
            format_type = export.get('format', 'unknown')
            formats_used[format_type] = formats_used.get(format_type, 0) + 1
            total_size += export.get('file_size', 0)

        return {
            'total_exports': len(export_history),
            'formats_used': formats_used,
            'total_size_mb': total_size / (1024 * 1024),
            'recent_exports': export_history[-5:],  # Last 5 exports
            'most_used_format': max(formats_used.items(), key=lambda x: x[1])[0] if formats_used else None
        }