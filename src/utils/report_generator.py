"""
Enhanced Analysis Report Generator
Generates comprehensive forensic analysis reports with recommendations and insights
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from collections import Counter, defaultdict
import logging

class ForensicReportGenerator:
    """
    Advanced report generator for forensic analysis results
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.report_data = {}
        self.recommendations = []
        self.executive_summary = ""
        
    def generate_comprehensive_report(self, 
                                    data: pd.DataFrame,
                                    anomalies: List,
                                    behavioral_patterns: List = None,
                                    analysis_type: str = "general",
                                    output_path: str = None) -> str:
        """
        Generate a comprehensive forensic analysis report
        
        Args:
            data: Original dataset
            anomalies: Detected anomalies
            behavioral_patterns: Behavioral analysis results
            analysis_type: Type of analysis (network, financial, general)
            output_path: Path to save the report
            
        Returns:
            Path to generated report file
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"forensic_analysis_report_{timestamp}.md"
        
        # Collect all analysis data
        self.report_data = {
            'data': data,
            'anomalies': anomalies,
            'behavioral_patterns': behavioral_patterns or [],
            'analysis_type': analysis_type,
            'timestamp': datetime.now(),
            'data_summary': self._analyze_data_quality(data),
            'anomaly_summary': self._analyze_anomalies(anomalies),
            'risk_assessment': self._assess_overall_risk(anomalies, behavioral_patterns or [])
        }
        
        # Generate report sections
        self._generate_executive_summary()
        self._generate_recommendations()
        
        # Write the complete report
        self._write_report_to_file(output_path)
        
        self.logger.info(f"Comprehensive report generated: {output_path}")
        return output_path
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        return {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'missing_data_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_records': data.duplicated().sum(),
            'data_types': data.dtypes.value_counts().to_dict(),
            'completeness_by_column': ((1 - data.isnull().sum() / len(data)) * 100).to_dict(),
            'numeric_columns': len(data.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns)
        }
    
    def _analyze_anomalies(self, anomalies: List) -> Dict[str, Any]:
        """Analyze anomaly patterns and statistics"""
        if not anomalies:
            return {
                'total_anomalies': 0,
                'severity_distribution': {},
                'confidence_stats': {},
                'anomaly_types': {},
                'high_priority_count': 0
            }
        
        # Extract severity distribution
        severities = [getattr(anomaly, 'severity', 'unknown') for anomaly in anomalies]
        severity_counts = Counter(severities)
        
        # Extract confidence statistics
        confidences = [getattr(anomaly, 'confidence', 0.0) for anomaly in anomalies]
        confidence_stats = {
            'mean': sum(confidences) / len(confidences) if confidences else 0,
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0,
            'std': pd.Series(confidences).std() if confidences else 0
        }
        
        # Extract anomaly types
        anomaly_types = [getattr(anomaly, 'anomaly_type', 'unknown') for anomaly in anomalies]
        type_counts = Counter(anomaly_types)
        
        # Count high priority anomalies
        high_priority_count = sum(1 for anomaly in anomalies 
                                if getattr(anomaly, 'severity', 'low') in ['high', 'critical'])
        
        return {
            'total_anomalies': len(anomalies),
            'severity_distribution': dict(severity_counts),
            'confidence_stats': confidence_stats,
            'anomaly_types': dict(type_counts.most_common(10)),
            'high_priority_count': high_priority_count
        }
    
    def _assess_overall_risk(self, anomalies: List, behavioral_patterns: List) -> Dict[str, Any]:
        """Assess overall risk level based on findings"""
        risk_score = 0
        risk_factors = []
        
        # Risk from anomalies
        if anomalies:
            critical_count = sum(1 for a in anomalies if getattr(a, 'severity', 'low') == 'critical')
            high_count = sum(1 for a in anomalies if getattr(a, 'severity', 'low') == 'high')
            
            risk_score += critical_count * 10 + high_count * 5
            
            if critical_count > 0:
                risk_factors.append(f"{critical_count} critical anomalies detected")
            if high_count > 0:
                risk_factors.append(f"{high_count} high-severity anomalies detected")
        
        # Risk from behavioral patterns
        if behavioral_patterns:
            critical_patterns = sum(1 for p in behavioral_patterns if getattr(p, 'severity', 'low') == 'critical')
            high_patterns = sum(1 for p in behavioral_patterns if getattr(p, 'severity', 'low') == 'high')
            
            risk_score += critical_patterns * 8 + high_patterns * 4
            
            if critical_patterns > 0:
                risk_factors.append(f"{critical_patterns} critical behavioral patterns")
            if high_patterns > 0:
                risk_factors.append(f"{high_patterns} high-risk behavioral patterns")
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "CRITICAL"
        elif risk_score >= 25:
            risk_level = "HIGH"
        elif risk_score >= 10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'immediate_action_required': risk_level in ['CRITICAL', 'HIGH']
        }
    
    def _generate_executive_summary(self):
        """Generate executive summary"""
        data_summary = self.report_data['data_summary']
        anomaly_summary = self.report_data['anomaly_summary']
        risk_assessment = self.report_data['risk_assessment']
        
        summary_lines = []
        
        # Overview
        summary_lines.append(f"This forensic analysis examined {data_summary['total_records']:,} records across {data_summary['total_columns']} data fields.")
        
        # Key findings
        if anomaly_summary['total_anomalies'] > 0:
            summary_lines.append(f"The analysis identified {anomaly_summary['total_anomalies']} anomalies, with {anomaly_summary['high_priority_count']} requiring immediate attention.")
        else:
            summary_lines.append("No significant anomalies were detected in the analyzed data.")
        
        # Risk assessment
        summary_lines.append(f"Overall risk level: {risk_assessment['risk_level']} (Score: {risk_assessment['risk_score']})")
        
        if risk_assessment['immediate_action_required']:
            summary_lines.append("IMMEDIATE ACTION REQUIRED: Critical security issues have been identified.")
        
        # Data quality
        if data_summary['missing_data_percentage'] > 10:
            summary_lines.append(f"Data quality concern: {data_summary['missing_data_percentage']:.1f}% missing data detected.")
        
        self.executive_summary = " ".join(summary_lines)
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        self.recommendations = []
        
        anomaly_summary = self.report_data['anomaly_summary']
        risk_assessment = self.report_data['risk_assessment']
        data_summary = self.report_data['data_summary']
        
        # Immediate actions for high-risk situations
        if risk_assessment['risk_level'] in ['CRITICAL', 'HIGH']:
            self.recommendations.append({
                'priority': 'IMMEDIATE',
                'category': 'Security Response',
                'action': 'Initiate incident response procedures for critical findings',
                'details': 'Review all critical and high-severity anomalies within 1 hour'
            })
        
        # Anomaly-specific recommendations
        if anomaly_summary['total_anomalies'] > 0:
            severity_dist = anomaly_summary['severity_distribution']
            
            if severity_dist.get('critical', 0) > 0:
                self.recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Investigation',
                    'action': f"Investigate {severity_dist['critical']} critical anomalies",
                    'details': 'These require immediate forensic investigation and potential containment'
                })
            
            if severity_dist.get('high', 0) > 0:
                self.recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Analysis',
                    'action': f"Analyze {severity_dist['high']} high-severity anomalies",
                    'details': 'Review within 24 hours and determine if escalation is needed'
                })
            
            if severity_dist.get('medium', 0) > 0:
                self.recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Monitoring',
                    'action': f"Monitor {severity_dist['medium']} medium-severity anomalies",
                    'details': 'Track patterns and escalate if frequency increases'
                })
        
        # Behavioral pattern recommendations
        if self.report_data['behavioral_patterns']:
            self.recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Behavioral Analysis',
                'action': 'Review detected behavioral patterns',
                'details': 'Analyze user and system behavior anomalies for potential policy violations'
            })
        
        # Data quality recommendations
        if data_summary['missing_data_percentage'] > 10:
            self.recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Quality',
                'action': 'Improve data collection processes',
                'details': f"Address {data_summary['missing_data_percentage']:.1f}% missing data to improve analysis accuracy"
            })
        
        if data_summary['duplicate_records'] > 0:
            self.recommendations.append({
                'priority': 'LOW',
                'category': 'Data Quality',
                'action': 'Implement data deduplication',
                'details': f"Remove {data_summary['duplicate_records']} duplicate records"
            })
        
        # General security recommendations
        self.recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Continuous Monitoring',
            'action': 'Implement ongoing monitoring',
            'details': 'Set up automated detection for identified anomaly patterns'
        })
        
        self.recommendations.append({
            'priority': 'LOW',
            'category': 'Documentation',
            'action': 'Update security policies',
            'details': 'Incorporate findings into security awareness and incident response procedures'
        })
    
    def _write_report_to_file(self, output_path: str):
        """Write the complete report to a markdown file"""
        
        # Start with the report header
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Forensic Analysis Report\n\n")
            
        # Add executive summary
        self._append_executive_summary(output_path)
        
        # Add key findings
        self._append_key_findings(output_path)
        
        # Add detailed analysis
        self._append_detailed_analysis(output_path)
        
        # Add recommendations
        self._append_recommendations(output_path)
        
        # Add technical details
        self._append_technical_details(output_path)
        
        # Add appendices
        self._append_appendices(output_path)
    
    def _append_executive_summary(self, output_path: str):
        """Append executive summary section"""
        content = f"""
## Executive Summary

**Analysis Date:** {self.report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Type:** {self.report_data['analysis_type'].title()}  
**Risk Level:** {self.report_data['risk_assessment']['risk_level']}

{self.executive_summary}

### Key Metrics
- **Total Records Analyzed:** {self.report_data['data_summary']['total_records']:,}
- **Anomalies Detected:** {self.report_data['anomaly_summary']['total_anomalies']}
- **High Priority Issues:** {self.report_data['anomaly_summary']['high_priority_count']}
- **Overall Risk Score:** {self.report_data['risk_assessment']['risk_score']}

"""
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _append_key_findings(self, output_path: str):
        """Append key findings section"""
        content = "## Key Findings\n\n"
        
        # Risk factors
        if self.report_data['risk_assessment']['risk_factors']:
            content += "### Critical Risk Factors\n"
            for factor in self.report_data['risk_assessment']['risk_factors']:
                content += f"- {factor}\n"
            content += "\n"
        
        # Top anomaly types
        anomaly_types = self.report_data['anomaly_summary']['anomaly_types']
        if anomaly_types:
            content += "### Most Common Anomaly Types\n"
            for anomaly_type, count in list(anomaly_types.items())[:5]:
                content += f"- **{anomaly_type}:** {count} occurrences\n"
            content += "\n"
        
        # Severity distribution
        severity_dist = self.report_data['anomaly_summary']['severity_distribution']
        if severity_dist:
            content += "### Severity Distribution\n"
            for severity, count in severity_dist.items():
                content += f"- **{severity.title()}:** {count} anomalies\n"
            content += "\n"
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _append_detailed_analysis(self, output_path: str):
        """Append detailed analysis section"""
        content = "## Detailed Analysis\n\n"
        
        # Data quality analysis
        content += "### Data Quality Assessment\n\n"
        data_summary = self.report_data['data_summary']
        
        content += f"**Dataset Overview:**\n"
        content += f"- Total records: {data_summary['total_records']:,}\n"
        content += f"- Total columns: {data_summary['total_columns']}\n"
        content += f"- Missing data: {data_summary['missing_data_percentage']:.2f}%\n"
        content += f"- Duplicate records: {data_summary['duplicate_records']}\n"
        content += f"- Numeric columns: {data_summary['numeric_columns']}\n"
        content += f"- Categorical columns: {data_summary['categorical_columns']}\n\n"
        
        # Anomaly analysis
        if self.report_data['anomaly_summary']['total_anomalies'] > 0:
            content += "### Anomaly Analysis\n\n"
            
            confidence_stats = self.report_data['anomaly_summary']['confidence_stats']
            content += f"**Confidence Statistics:**\n"
            content += f"- Average confidence: {confidence_stats['mean']:.3f}\n"
            content += f"- Confidence range: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}\n"
            content += f"- Standard deviation: {confidence_stats['std']:.3f}\n\n"
        
        # Behavioral patterns
        if self.report_data['behavioral_patterns']:
            content += "### Behavioral Pattern Analysis\n\n"
            
            pattern_types = Counter(getattr(p, 'pattern_type', 'Unknown') for p in self.report_data['behavioral_patterns'])
            content += "**Detected Pattern Types:**\n"
            for pattern_type, count in pattern_types.most_common():
                content += f"- {pattern_type}: {count} instances\n"
            content += "\n"
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _append_recommendations(self, output_path: str):
        """Append recommendations section"""
        content = "## Recommendations\n\n"
        
        # Group recommendations by priority
        priority_groups = defaultdict(list)
        for rec in self.recommendations:
            priority_groups[rec['priority']].append(rec)
        
        # Write recommendations by priority
        for priority in ['IMMEDIATE', 'HIGH', 'MEDIUM', 'LOW']:
            if priority in priority_groups:
                content += f"### {priority} Priority\n\n"
                
                for rec in priority_groups[priority]:
                    content += f"**{rec['category']}: {rec['action']}**\n"
                    content += f"{rec['details']}\n\n"
        
        # Implementation timeline
        content += "### Implementation Timeline\n\n"
        content += "- **Immediate (0-1 hours):** Address all immediate priority items\n"
        content += "- **High Priority (1-24 hours):** Complete high priority investigations\n"
        content += "- **Medium Priority (1-7 days):** Implement medium priority improvements\n"
        content += "- **Low Priority (1-30 days):** Address low priority enhancements\n\n"
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _append_technical_details(self, output_path: str):
        """Append technical details section"""
        content = "## Technical Details\n\n"
        
        # Analysis methodology
        content += "### Analysis Methodology\n\n"
        content += "This forensic analysis employed multiple detection techniques:\n\n"
        content += "- **Statistical Analysis:** Outlier detection using IQR and Z-score methods\n"
        content += "- **Machine Learning:** Ensemble methods including Random Forest, XGBoost, and Isolation Forest\n"
        content += "- **Behavioral Analysis:** Pattern recognition for user and system behaviors\n"
        content += "- **Temporal Analysis:** Time-based anomaly detection\n"
        content += "- **Correlation Analysis:** Cross-feature relationship analysis\n\n"
        
        # Detection thresholds
        content += "### Detection Thresholds\n\n"
        content += "- **Critical:** Confidence > 0.9, High impact potential\n"
        content += "- **High:** Confidence > 0.8, Significant risk indicators\n"
        content += "- **Medium:** Confidence > 0.6, Moderate risk patterns\n"
        content += "- **Low:** Confidence > 0.4, Minor anomalies for monitoring\n\n"
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _append_appendices(self, output_path: str):
        """Append appendices section"""
        content = "## Appendices\n\n"
        
        # Appendix A: Data completeness by column
        content += "### Appendix A: Data Completeness by Column\n\n"
        completeness = self.report_data['data_summary']['completeness_by_column']
        
        content += "| Column | Completeness |\n"
        content += "|--------|-------------|\n"
        
        for column, percentage in sorted(completeness.items(), key=lambda x: x[1]):
            content += f"| {column} | {percentage:.1f}% |\n"
        
        content += "\n"
        
        # Appendix B: Anomaly details (top 10)
        if self.report_data['anomalies']:
            content += "### Appendix B: Top Anomalies (Sample)\n\n"
            content += "| Type | Severity | Confidence | Description |\n"
            content += "|------|----------|------------|-------------|\n"
            
            # Sort anomalies by severity and confidence
            sorted_anomalies = sorted(
                self.report_data['anomalies'][:10],  # Top 10
                key=lambda x: (
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(getattr(x, 'severity', 'low'), 0),
                    getattr(x, 'confidence', 0)
                ),
                reverse=True
            )
            
            for anomaly in sorted_anomalies:
                anomaly_type = getattr(anomaly, 'anomaly_type', 'Unknown')
                severity = getattr(anomaly, 'severity', 'Unknown')
                confidence = getattr(anomaly, 'confidence', 0)
                description = getattr(anomaly, 'description', 'No description')
                
                # Truncate long descriptions
                if len(description) > 50:
                    description = description[:47] + "..."
                
                content += f"| {anomaly_type} | {severity} | {confidence:.3f} | {description} |\n"
            
            content += "\n"
        
        # Appendix C: Report metadata
        content += "### Appendix C: Report Metadata\n\n"
        content += f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"- **Analysis Duration:** {self.report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"- **Report Version:** 1.0\n"
        content += f"- **Tool:** CyberForensics Data Detective\n\n"
        
        content += "---\n"
        content += "*This report was generated automatically by the CyberForensics Data Detective analysis engine.*\n"
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def generate_json_report(self, output_path: str = None) -> str:
        """Generate a machine-readable JSON report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"forensic_analysis_data_{timestamp}.json"
        
        # Prepare JSON-serializable data
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        json_data = {
            'metadata': {
                'generated_at': self.report_data['timestamp'].isoformat(),
                'analysis_type': self.report_data['analysis_type'],
                'report_version': '1.0'
            },
            'data_summary': convert_to_serializable(self.report_data['data_summary']),
            'anomaly_summary': convert_to_serializable(self.report_data['anomaly_summary']),
            'risk_assessment': convert_to_serializable(self.report_data['risk_assessment']),
            'executive_summary': self.executive_summary,
            'recommendations': self.recommendations,
            'anomalies': [],
            'behavioral_patterns': []
        }
        
        # Add anomaly details
        for anomaly in self.report_data['anomalies'][:100]:  # Limit to 100 for file size
            anomaly_dict = {
                'type': getattr(anomaly, 'anomaly_type', 'Unknown'),
                'severity': getattr(anomaly, 'severity', 'Unknown'),
                'confidence': float(getattr(anomaly, 'confidence', 0)),
                'description': getattr(anomaly, 'description', ''),
                'timestamp': getattr(anomaly, 'timestamp', datetime.now()).isoformat() if hasattr(anomaly, 'timestamp') else None
            }
            json_data['anomalies'].append(anomaly_dict)
        
        # Add behavioral pattern details
        for pattern in self.report_data['behavioral_patterns'][:50]:  # Limit to 50
            pattern_dict = {
                'type': getattr(pattern, 'pattern_type', 'Unknown'),
                'severity': getattr(pattern, 'severity', 'Unknown'),
                'confidence': float(getattr(pattern, 'confidence', 0)),
                'description': getattr(pattern, 'description', ''),
                'entities': getattr(pattern, 'entities', [])
            }
            json_data['behavioral_patterns'].append(pattern_dict)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report generated: {output_path}")
        return output_path

# Global instance for easy access
report_generator = ForensicReportGenerator()

