"""
Export manager for CyberForensics Data Detective.
Handles exporting analysis results, reports, and visualizations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from utils.helpers import sanitize_filename, create_report_filename


class ExportManager:
    """Manager for exporting analysis results and reports."""
    
    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.supported_formats = {
            'csv': self._export_csv,
            'json': self._export_json,
            'excel': self._export_excel,
            'html': self._export_html,
            'txt': self._export_text
        }
    
    def export_analysis_results(self, 
                               results: Dict[str, Any], 
                               analysis_type: str,
                               format: str = 'csv',
                               filename: Optional[str] = None) -> Path:
        """Export analysis results to specified format."""
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.supported_formats.keys())}")
        
        if filename is None:
            filename = create_report_filename(f"{analysis_type}_analysis", datetime.now())
            filename = filename.replace('.csv', f'.{format}')
        
        filename = sanitize_filename(filename)
        output_path = self.output_dir / filename
        
        self.logger.info(f"Exporting {analysis_type} results to {output_path}")
        
        # Prepare data for export
        export_data = self._prepare_export_data(results, analysis_type)
        
        # Export using appropriate method
        self.supported_formats[format](export_data, output_path)
        
        self.logger.info(f"Export completed: {output_path}")
        return output_path
    
    def export_dataframe(self, 
                        df: pd.DataFrame, 
                        filename: str,
                        format: str = 'csv',
                        include_metadata: bool = True) -> Path:
        """Export DataFrame to specified format."""
        
        filename = sanitize_filename(filename)
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        output_path = self.output_dir / filename
        
        self.logger.info(f"Exporting DataFrame to {output_path}")
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported DataFrame export format: {format}")
        
        # Add metadata file if requested
        if include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return output_path
    
    def create_investigation_report(self, 
                                  investigation_data: Dict[str, Any],
                                  template: str = 'standard') -> Path:
        """Create a comprehensive investigation report."""
        
        timestamp = datetime.now()
        filename = f"investigation_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_dir / filename
        
        if template == 'standard':
            html_content = self._create_standard_report(investigation_data, timestamp)
        elif template == 'executive':
            html_content = self._create_executive_report(investigation_data, timestamp)
        else:
            raise ValueError(f"Unknown report template: {template}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Investigation report created: {output_path}")
        return output_path
    
    def export_timeline(self, timeline_data: List[Dict[str, Any]], filename: Optional[str] = None) -> Path:
        """Export timeline data for forensic analysis."""
        
        if filename is None:
            filename = create_report_filename("timeline", datetime.now())
        
        output_path = self.output_dir / sanitize_filename(filename)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(timeline_data)
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Export as CSV with timeline-specific formatting
        df.to_csv(output_path, index=False)
        
        # Also create a human-readable timeline
        timeline_txt_path = output_path.with_suffix('.timeline.txt')
        with open(timeline_txt_path, 'w', encoding='utf-8') as f:
            f.write("FORENSIC TIMELINE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for _, event in df.iterrows():
                timestamp = event.get('timestamp', 'Unknown')
                event_type = event.get('event_type', 'Unknown')
                description = event.get('description', 'No description')
                
                f.write(f"{timestamp} | {event_type} | {description}\n")
        
        return output_path
    
    def _prepare_export_data(self, results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Prepare results data for export."""
        
        export_data = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        # Add summary statistics if available
        if 'summary' in results:
            export_data['summary'] = results['summary']
        
        # Convert DataFrames to dictionaries for JSON export
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                export_data['results'][key] = {
                    'data': value.to_dict('records'),
                    'shape': value.shape,
                    'columns': list(value.columns)
                }
        
        return export_data
    
    def _export_csv(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as CSV."""
        # If results contain DataFrames, export the main one
        results = data.get('results', {})
        
        # Find the main DataFrame to export
        main_df = None
        for key, value in results.items():
            if isinstance(value, dict) and 'data' in value:
                main_df = pd.DataFrame(value['data'])
                break
            elif isinstance(value, pd.DataFrame):
                main_df = value
                break
        
        if main_df is not None:
            main_df.to_csv(output_path, index=False)
        else:
            # Export as flattened CSV
            flattened_data = self._flatten_dict(data)
            df = pd.DataFrame([flattened_data])
            df.to_csv(output_path, index=False)
    
    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_excel(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as Excel with multiple sheets."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Analysis Type': [data.get('analysis_type', 'Unknown')],
                    'Timestamp': [data.get('timestamp', 'Unknown')],
                    'Export Date': [datetime.now().isoformat()]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Results sheets
                results = data.get('results', {})
                for key, value in results.items():
                    if isinstance(value, dict) and 'data' in value:
                        df = pd.DataFrame(value['data'])
                        sheet_name = key[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif isinstance(value, pd.DataFrame):
                        sheet_name = key[:31]
                        value.to_excel(writer, sheet_name=sheet_name, index=False)
        
        except ImportError:
            self.logger.warning("openpyxl not available, falling back to CSV export")
            csv_path = output_path.with_suffix('.csv')
            self._export_csv(data, csv_path)
    
    def _export_html(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as HTML report."""
        html_content = self._create_html_report(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_text(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as plain text."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"CyberForensics Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Type: {data.get('analysis_type', 'Unknown')}\n")
            f.write(f"Timestamp: {data.get('timestamp', 'Unknown')}\n\n")
            
            results = data.get('results', {})
            for key, value in results.items():
                f.write(f"{key.upper()}:\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        f.write(f"{sub_key}: {sub_value}\n")
                else:
                    f.write(f"{value}\n")
                
                f.write("\n")
    
    def _create_html_report(self, data: Dict[str, Any]) -> str:
        """Create HTML report from data."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CyberForensics Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .summary {{ background-color: #ecf0f1; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CyberForensics Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section summary">
                <h2>Analysis Summary</h2>
                <p><strong>Type:</strong> {data.get('analysis_type', 'Unknown')}</p>
                <p><strong>Timestamp:</strong> {data.get('timestamp', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Results</h2>
                {self._format_results_html(data.get('results', {}))}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_results_html(self, results: Dict[str, Any]) -> str:
        """Format results as HTML."""
        html_parts = []
        
        for key, value in results.items():
            html_parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>")
            
            if isinstance(value, dict) and 'data' in value:
                # DataFrame data
                df = pd.DataFrame(value['data'])
                if len(df) > 0:
                    html_parts.append(df.to_html(classes='table', escape=False))
                else:
                    html_parts.append("<p>No data available</p>")
            elif isinstance(value, pd.DataFrame):
                html_parts.append(value.to_html(classes='table', escape=False))
            elif isinstance(value, dict):
                html_parts.append("<ul>")
                for sub_key, sub_value in value.items():
                    html_parts.append(f"<li><strong>{sub_key}:</strong> {sub_value}</li>")
                html_parts.append("</ul>")
            else:
                html_parts.append(f"<p>{value}</p>")
        
        return "\n".join(html_parts)
    
    def _create_standard_report(self, investigation_data: Dict[str, Any], timestamp: datetime) -> str:
        """Create standard investigation report."""
        # This would be a comprehensive template for investigation reports
        # For now, return a basic structure
        return self._create_html_report(investigation_data)
    
    def _create_executive_report(self, investigation_data: Dict[str, Any], timestamp: datetime) -> str:
        """Create executive summary report."""
        # This would be a high-level summary template
        # For now, return a basic structure
        return self._create_html_report(investigation_data)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_export_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return list(self.supported_formats.keys())
    
    def set_output_directory(self, output_dir: Union[str, Path]) -> None:
        """Set the output directory for exports."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Output directory set to: {self.output_dir}")

