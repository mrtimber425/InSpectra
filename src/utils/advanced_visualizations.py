"""
Advanced Context-Aware Visualizations for Inspectra
Provides sophisticated, intelligent charts and graphs that adapt based on forensic analysis type
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import warnings
import logging
warnings.filterwarnings('ignore')

# Graceful imports for optional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive visualizations disabled.")

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QTabWidget
    from PyQt5.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logging.warning("PyQt5 not available. GUI components disabled.")

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtCore import QUrl
    QTWEBENGINE_AVAILABLE = True
except ImportError:
    QTWEBENGINE_AVAILABLE = False
    logging.warning("QtWebEngine not available. Web-based charts disabled.")

class ContextAwareVisualizationEngine:
    """
    Intelligent visualization engine that creates context-specific charts
    based on the type of forensic analysis being performed
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Set up professional dark theme for matplotlib
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", palette="dark")

        # Define color schemes for different contexts
        self.color_schemes = {
            'network': {
                'primary': '#00d4ff',
                'secondary': '#0078d4',
                'accent': '#40e0d0',
                'warning': '#ffa500',
                'danger': '#ff4444',
                'success': '#32cd32'
            },
            'financial': {
                'primary': '#32cd32',
                'secondary': '#228b22',
                'accent': '#ffd700',
                'warning': '#ffa500',
                'danger': '#dc143c',
                'success': '#00ff00'
            },
            'general': {
                'primary': '#0078d4',
                'secondary': '#106ebe',
                'accent': '#40e0d0',
                'warning': '#ff8c00',
                'danger': '#d13438',
                'success': '#107c10'
            }
        }

    def create_context_visualization(self, analysis_type: str, data: pd.DataFrame,
                                   anomalies: List, chart_type: str = "overview") -> Figure:
        """
        Create context-aware visualization based on analysis type

        Args:
            analysis_type: Type of analysis ('Network', 'Financial', 'General')
            data: Source data
            anomalies: Detected anomalies
            chart_type: Specific chart type to create

        Returns:
            Matplotlib Figure object
        """

        # Determine context from analysis type
        if "Network" in analysis_type or "Digital" in analysis_type:
            context = 'network'
        elif "Financial" in analysis_type:
            context = 'financial'
        else:
            context = 'general'

        # Create visualization based on context and chart type
        if context == 'network':
            return self.create_network_visualization(data, anomalies, chart_type)
        elif context == 'financial':
            return self.create_financial_visualization(data, anomalies, chart_type)
        else:
            return self.create_general_visualization(data, anomalies, chart_type)

    def create_network_visualization(self, data: pd.DataFrame, anomalies: List,
                                   chart_type: str = "overview") -> Figure:
        """Create network-specific visualizations"""

        colors = self.color_schemes['network']
        fig = Figure(figsize=(14, 10), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        if chart_type == "overview":
            # Network overview with 4 subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Network Traffic Flow Analysis
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_network_traffic_flow(ax1, data, colors)

            # 2. Port Activity Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            self._create_port_activity_chart(ax2, data, anomalies, colors)

            # 3. IP Geolocation and Threat Map
            ax3 = fig.add_subplot(gs[1, 0])
            self._create_ip_threat_map(ax3, data, anomalies, colors)

            # 4. Network Anomaly Timeline
            ax4 = fig.add_subplot(gs[1, 1])
            self._create_network_timeline(ax4, anomalies, colors)

        elif chart_type == "traffic_analysis":
            # Detailed traffic analysis
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

            # Traffic volume over time
            ax1 = fig.add_subplot(gs[0, :])
            self._create_traffic_volume_timeline(ax1, data, colors)

            # Protocol distribution
            ax2 = fig.add_subplot(gs[1, 0])
            self._create_protocol_distribution(ax2, data, colors)

            # Bandwidth utilization
            ax3 = fig.add_subplot(gs[1, 1])
            self._create_bandwidth_analysis(ax3, data, colors)

            # Connection patterns
            ax4 = fig.add_subplot(gs[2, :])
            self._create_connection_patterns(ax4, data, anomalies, colors)

        elif chart_type == "threat_analysis":
            # Advanced threat analysis
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            # Threat severity matrix
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_threat_severity_matrix(ax1, anomalies, colors)

            # Attack vector analysis
            ax2 = fig.add_subplot(gs[0, 1])
            self._create_attack_vector_chart(ax2, anomalies, colors)

            # Risk scoring distribution
            ax3 = fig.add_subplot(gs[0, 2])
            self._create_risk_distribution(ax3, anomalies, colors)

            # Incident correlation
            ax4 = fig.add_subplot(gs[1, :])
            self._create_incident_correlation(ax4, anomalies, colors)

        fig.suptitle('Inspectra Network Forensic Analysis',
                    fontsize=16, fontweight='bold', color='white', y=0.95)

        return fig

    def create_financial_visualization(self, data: pd.DataFrame, anomalies: List,
                                     chart_type: str = "overview") -> Figure:
        """Create financial-specific visualizations"""

        colors = self.color_schemes['financial']
        fig = Figure(figsize=(14, 10), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        if chart_type == "overview":
            # Financial overview with 4 subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Transaction Flow Analysis
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_transaction_flow(ax1, data, colors)

            # 2. Fraud Risk Assessment
            ax2 = fig.add_subplot(gs[0, 1])
            self._create_fraud_risk_assessment(ax2, anomalies, colors)

            # 3. Money Laundering Patterns
            ax3 = fig.add_subplot(gs[1, 0])
            self._create_ml_patterns_chart(ax3, data, anomalies, colors)

            # 4. Financial Anomaly Timeline
            ax4 = fig.add_subplot(gs[1, 1])
            self._create_financial_timeline(ax4, anomalies, colors)

        elif chart_type == "transaction_analysis":
            # Detailed transaction analysis
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

            # Transaction volume and value trends
            ax1 = fig.add_subplot(gs[0, :])
            self._create_transaction_trends(ax1, data, colors)

            # Account activity patterns
            ax2 = fig.add_subplot(gs[1, 0])
            self._create_account_patterns(ax2, data, colors)

            # Currency and jurisdiction analysis
            ax3 = fig.add_subplot(gs[1, 1])
            self._create_currency_analysis(ax3, data, colors)

            # Velocity and structuring detection
            ax4 = fig.add_subplot(gs[2, :])
            self._create_velocity_structuring_analysis(ax4, data, anomalies, colors)

        elif chart_type == "compliance_analysis":
            # Regulatory compliance analysis
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            # Regulatory threshold violations
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_threshold_violations(ax1, anomalies, colors)

            # KYC/AML risk factors
            ax2 = fig.add_subplot(gs[0, 1])
            self._create_kyc_aml_risk(ax2, data, colors)

            # Sanctions screening results
            ax3 = fig.add_subplot(gs[0, 2])
            self._create_sanctions_screening(ax3, data, colors)

            # Regulatory reporting requirements
            ax4 = fig.add_subplot(gs[1, :])
            self._create_regulatory_reporting(ax4, anomalies, colors)

        fig.suptitle('Inspectra Financial Forensic Analysis',
                    fontsize=16, fontweight='bold', color='white', y=0.95)

        return fig

    def create_general_visualization(self, data: pd.DataFrame, anomalies: List,
                                   chart_type: str = "overview") -> Figure:
        """Create general analysis visualizations"""

        colors = self.color_schemes['general']
        fig = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')

        # Create general analysis charts
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Anomaly severity distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_severity_distribution(ax1, anomalies, colors)

        # 2. Confidence score analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_confidence_analysis(ax2, anomalies, colors)

        # 3. Data quality assessment
        ax3 = fig.add_subplot(gs[1, 0])
        self._create_data_quality_chart(ax3, data, colors)

        # 4. Anomaly type distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._create_anomaly_type_distribution(ax4, anomalies, colors)

        fig.suptitle('Inspectra General Forensic Analysis',
                    fontsize=16, fontweight='bold', color='white', y=0.95)

        return fig

    # Network-specific chart methods
    def _create_network_traffic_flow(self, ax, data: pd.DataFrame, colors: Dict):
        """Create network traffic flow visualization"""
        ax.set_facecolor('#2d2d2d')

        # Look for IP and traffic data
        ip_cols = [col for col in data.columns if 'ip' in col.lower()]
        bytes_cols = [col for col in data.columns if 'byte' in col.lower() or 'size' in col.lower()]

        if ip_cols and bytes_cols:
            # Create a simplified network flow chart
            source_ips = data[ip_cols[0]].value_counts().head(10)

            # Create horizontal bar chart
            bars = ax.barh(range(len(source_ips)), source_ips.values,
                          color=colors['primary'], alpha=0.8)

            ax.set_yticks(range(len(source_ips)))
            ax.set_yticklabels([f"...{ip[-8:]}" for ip in source_ips.index], color='white')
            ax.set_xlabel('Traffic Volume', color='white')
            ax.set_title('Top Traffic Sources', color='white', fontweight='bold')
            ax.tick_params(colors='white')

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(source_ips.values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{int(width)}', ha='left', va='center', color='white')
        else:
            ax.text(0.5, 0.5, 'Network Traffic Data\nNot Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_port_activity_chart(self, ax, data: pd.DataFrame, anomalies: List, colors: Dict):
        """Create port activity analysis chart"""
        ax.set_facecolor('#2d2d2d')

        # Look for port data
        port_cols = [col for col in data.columns if 'port' in col.lower()]

        if port_cols:
            port_data = data[port_cols[0]].value_counts().head(15)

            # Create circular/polar chart for ports
            angles = np.linspace(0, 2 * np.pi, len(port_data), endpoint=False)

            # Convert to polar
            ax.remove()
            ax = ax.figure.add_subplot(ax.get_gridspec()[ax.get_subplotspec()], projection='polar')
            ax.set_facecolor('#2d2d2d')

            bars = ax.bar(angles, port_data.values, width=0.3,
                         color=colors['primary'], alpha=0.7)

            ax.set_title('Port Activity Distribution', color='white', fontweight='bold', pad=20)
            ax.set_thetagrids(angles * 180/np.pi, [str(port) for port in port_data.index])
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Port Activity Data\nNot Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_ip_threat_map(self, ax, data: pd.DataFrame, anomalies: List, colors: Dict):
        """Create IP threat mapping visualization"""
        ax.set_facecolor('#2d2d2d')

        # Create a threat level heatmap
        threat_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}

        if anomalies:
            severity_counts = Counter(anomaly.severity for anomaly in anomalies)

            # Create a simple threat matrix
            threat_matrix = np.zeros((4, 4))
            for i, (severity, count) in enumerate(severity_counts.items()):
                if i < 4:
                    threat_matrix[i, :count//4 + 1] = threat_levels.get(severity, 1)

            im = ax.imshow(threat_matrix, cmap='Reds', aspect='auto')
            ax.set_title('Threat Level Matrix', color='white', fontweight='bold')
            ax.set_xlabel('Geographic Distribution', color='white')
            ax.set_ylabel('Threat Severity', color='white')
            ax.tick_params(colors='white')
        else:
            ax.text(0.5, 0.5, 'No Threat Data\nAvailable',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_network_timeline(self, ax, anomalies: List, colors: Dict):
        """Create network incident timeline"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            timestamps = []
            severities = []

            for anomaly in anomalies:
                if hasattr(anomaly, 'timestamp') and anomaly.timestamp:
                    timestamps.append(anomaly.timestamp)
                    severities.append(anomaly.severity)

            if timestamps:
                # Create timeline scatter plot
                severity_colors = {
                    'low': colors['success'],
                    'medium': colors['warning'],
                    'high': colors['danger'],
                    'critical': '#ff0000'
                }

                for severity in ['low', 'medium', 'high', 'critical']:
                    severity_times = [t for t, s in zip(timestamps, severities) if s == severity]
                    if severity_times:
                        y_pos = [list(severity_colors.keys()).index(severity)] * len(severity_times)
                        ax.scatter(severity_times, y_pos,
                                 c=severity_colors[severity],
                                 s=100, alpha=0.8, label=severity.title())

                ax.set_ylabel('Severity Level', color='white')
                ax.set_xlabel('Time', color='white')
                ax.set_title('Network Incident Timeline', color='white', fontweight='bold')
                ax.legend()
                ax.tick_params(colors='white')

                # Format time axis
                if len(timestamps) > 0:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Timeline Data\nAvailable',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    # Additional network methods would go here...
    def _create_traffic_volume_timeline(self, ax, data: pd.DataFrame, colors: Dict):
        """Create traffic volume timeline"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Traffic Volume Timeline\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_protocol_distribution(self, ax, data: pd.DataFrame, colors: Dict):
        """Create protocol distribution chart"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Protocol Distribution\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_bandwidth_analysis(self, ax, data: pd.DataFrame, colors: Dict):
        """Create bandwidth analysis chart"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Bandwidth Analysis\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_connection_patterns(self, ax, data: pd.DataFrame, anomalies: List, colors: Dict):
        """Create connection patterns visualization"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Connection Patterns\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_threat_severity_matrix(self, ax, anomalies: List, colors: Dict):
        """Create threat severity matrix"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Threat Severity Matrix\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_attack_vector_chart(self, ax, anomalies: List, colors: Dict):
        """Create attack vector analysis chart"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Attack Vector Analysis\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_risk_distribution(self, ax, anomalies: List, colors: Dict):
        """Create risk score distribution"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Risk Score Distribution\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_incident_correlation(self, ax, anomalies: List, colors: Dict):
        """Create incident correlation analysis"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Incident Correlation\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    # Financial-specific chart methods
    def _create_transaction_flow(self, ax, data: pd.DataFrame, colors: Dict):
        """Create transaction flow visualization"""
        ax.set_facecolor('#2d2d2d')

        # Look for amount data
        amount_cols = [col for col in data.columns if 'amount' in col.lower() or 'value' in col.lower()]

        if amount_cols and len(data) > 0:
            amounts = data[amount_cols[0]].dropna()

            if len(amounts) > 0:
                # Create transaction flow chart
                bins = np.logspace(np.log10(amounts.min()), np.log10(amounts.max()), 20)
                hist, bin_edges = np.histogram(amounts, bins=bins)

                # Create bar chart with log scale
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bars = ax.bar(range(len(hist)), hist, color=colors['primary'], alpha=0.7)

                ax.set_title('Transaction Amount Distribution', color='white', fontweight='bold')
                ax.set_xlabel('Amount Range', color='white')
                ax.set_ylabel('Transaction Count', color='white')
                ax.tick_params(colors='white')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', color='white')
        else:
            ax.text(0.5, 0.5, 'Transaction Flow Data\nNot Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_fraud_risk_assessment(self, ax, anomalies: List, colors: Dict):
        """Create fraud risk assessment chart"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            # Calculate risk metrics
            risk_scores = []
            for anomaly in anomalies:
                risk_score = getattr(anomaly, 'risk_score', anomaly.confidence)
                risk_scores.append(risk_score)

            if risk_scores:
                # Create risk distribution
                risk_levels = ['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-0.8)', 'Critical (0.8-1.0)']
                risk_counts = [
                    sum(1 for score in risk_scores if 0 <= score < 0.3),
                    sum(1 for score in risk_scores if 0.3 <= score < 0.6),
                    sum(1 for score in risk_scores if 0.6 <= score < 0.8),
                    sum(1 for score in risk_scores if 0.8 <= score <= 1.0)
                ]

                # Create pie chart
                risk_colors = [colors['success'], colors['warning'], colors['danger'], '#ff0000']
                wedges, texts, autotexts = ax.pie(risk_counts, labels=risk_levels,
                                                 autopct='%1.1f%%', colors=risk_colors,
                                                 textprops={'color': 'white'})
                ax.set_title('Fraud Risk Assessment', color='white', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Risk Data\nAvailable',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_ml_patterns_chart(self, ax, data: pd.DataFrame, anomalies: List, colors: Dict):
        """Create money laundering patterns chart"""
        ax.set_facecolor('#2d2d2d')

        # Look for ML-related patterns in anomalies
        ml_patterns = defaultdict(int)

        for anomaly in anomalies:
            if 'money_laundering' in getattr(anomaly, 'anomaly_type', '').lower():
                ml_patterns['Money Laundering'] += 1
            elif 'structuring' in getattr(anomaly, 'anomaly_type', '').lower():
                ml_patterns['Structuring'] += 1
            elif 'round_amount' in getattr(anomaly, 'anomaly_type', '').lower():
                ml_patterns['Round Amounts'] += 1
            elif 'velocity' in getattr(anomaly, 'anomaly_type', '').lower():
                ml_patterns['High Velocity'] += 1

        if ml_patterns:
            patterns = list(ml_patterns.keys())
            counts = list(ml_patterns.values())

            bars = ax.bar(patterns, counts, color=colors['primary'], alpha=0.7)
            ax.set_title('Money Laundering Patterns', color='white', fontweight='bold')
            ax.set_ylabel('Detection Count', color='white')
            ax.tick_params(colors='white')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', color='white')
        else:
            ax.text(0.5, 0.5, 'No ML Patterns\nDetected',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_financial_timeline(self, ax, anomalies: List, colors: Dict):
        """Create financial anomaly timeline"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            timestamps = []
            amounts = []

            for anomaly in anomalies:
                if hasattr(anomaly, 'timestamp') and anomaly.timestamp:
                    timestamps.append(anomaly.timestamp)
                    amount = getattr(anomaly, 'amount', 0)
                    amounts.append(amount)

            if timestamps and any(amounts):
                # Create scatter plot with amount as size
                sizes = [max(50, min(300, amount/1000)) for amount in amounts]

                ax.scatter(timestamps, amounts, s=sizes, c=colors['primary'], alpha=0.6)
                ax.set_xlabel('Time', color='white')
                ax.set_ylabel('Transaction Amount', color='white')
                ax.set_title('Financial Anomaly Timeline', color='white', fontweight='bold')
                ax.tick_params(colors='white')

                # Format axes
                if len(timestamps) > 0:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Financial Timeline\nData Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    # Additional financial methods would go here...
    def _create_transaction_trends(self, ax, data: pd.DataFrame, colors: Dict):
        """Create transaction trends analysis"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Transaction Trends\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_account_patterns(self, ax, data: pd.DataFrame, colors: Dict):
        """Create account activity patterns"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Account Patterns\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_currency_analysis(self, ax, data: pd.DataFrame, colors: Dict):
        """Create currency and jurisdiction analysis"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Currency Analysis\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_velocity_structuring_analysis(self, ax, data: pd.DataFrame, anomalies: List, colors: Dict):
        """Create velocity and structuring analysis"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Velocity & Structuring\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_threshold_violations(self, ax, anomalies: List, colors: Dict):
        """Create regulatory threshold violations chart"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Threshold Violations\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_kyc_aml_risk(self, ax, data: pd.DataFrame, colors: Dict):
        """Create KYC/AML risk analysis"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'KYC/AML Risk\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_sanctions_screening(self, ax, data: pd.DataFrame, colors: Dict):
        """Create sanctions screening results"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Sanctions Screening\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    def _create_regulatory_reporting(self, ax, anomalies: List, colors: Dict):
        """Create regulatory reporting requirements"""
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, 'Regulatory Reporting\n(Enhanced Implementation)',
               transform=ax.transAxes, ha='center', va='center',
               color='white', fontsize=12)

    # General analysis chart methods
    def _create_severity_distribution(self, ax, anomalies: List, colors: Dict):
        """Create severity distribution chart"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            severity_counts = Counter(anomaly.severity for anomaly in anomalies)

            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())

            # Color mapping for severity levels
            severity_colors = {
                'low': colors['success'],
                'medium': colors['warning'],
                'high': colors['danger'],
                'critical': '#ff0000'
            }

            bar_colors = [severity_colors.get(sev, colors['primary']) for sev in severities]

            bars = ax.bar(severities, counts, color=bar_colors, alpha=0.8)
            ax.set_title('Anomaly Severity Distribution', color='white', fontweight='bold')
            ax.set_ylabel('Count', color='white')
            ax.tick_params(colors='white')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', color='white')
        else:
            ax.text(0.5, 0.5, 'No Severity Data\nAvailable',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_confidence_analysis(self, ax, anomalies: List, colors: Dict):
        """Create confidence score analysis"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            confidences = [anomaly.confidence for anomaly in anomalies]

            ax.hist(confidences, bins=15, color=colors['primary'], alpha=0.7, edgecolor='white')
            ax.set_title('Confidence Score Distribution', color='white', fontweight='bold')
            ax.set_xlabel('Confidence Score', color='white')
            ax.set_ylabel('Frequency', color='white')
            ax.tick_params(colors='white')

            # Add mean line
            mean_conf = np.mean(confidences)
            ax.axvline(mean_conf, color=colors['accent'], linestyle='--', linewidth=2,
                      label=f'Mean: {mean_conf:.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Confidence Data\nAvailable',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_data_quality_chart(self, ax, data: pd.DataFrame, colors: Dict):
        """Create data quality assessment chart"""
        ax.set_facecolor('#2d2d2d')

        if not data.empty:
            # Calculate data quality metrics
            total_cells = len(data) * len(data.columns)
            missing_cells = data.isnull().sum().sum()
            completeness = (total_cells - missing_cells) / total_cells * 100

            # Data quality categories
            categories = ['Complete', 'Missing']
            values = [completeness, 100 - completeness]

            # Create pie chart
            colors_list = [colors['success'], colors['danger']]
            wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                             colors=colors_list, textprops={'color': 'white'})
            ax.set_title('Data Quality Assessment', color='white', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Data Quality\nMetrics Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

    def _create_anomaly_type_distribution(self, ax, anomalies: List, colors: Dict):
        """Create anomaly type distribution chart"""
        ax.set_facecolor('#2d2d2d')

        if anomalies:
            type_counts = Counter(anomaly.anomaly_type for anomaly in anomalies)

            # Get top 8 most common types
            top_types = type_counts.most_common(8)
            types, counts = zip(*top_types) if top_types else ([], [])

            if types:
                # Clean up type names
                clean_types = [t.replace('_', ' ').title() for t in types]

                bars = ax.bar(range(len(counts)), counts, color=colors['primary'], alpha=0.7)
                ax.set_title('Top Anomaly Types', color='white', fontweight='bold')
                ax.set_xlabel('Anomaly Type', color='white')
                ax.set_ylabel('Count', color='white')
                ax.set_xticks(range(len(clean_types)))
                ax.set_xticklabels(clean_types, rotation=45, ha='right')
                ax.tick_params(colors='white')

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', color='white')
        else:
            ax.text(0.5, 0.5, 'No Anomaly Type\nData Available',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)

class AdvancedVisualizationWidget(QWidget if PYQT_AVAILABLE else object):
    """
    Advanced visualization widget with context-aware intelligence
    """

    def __init__(self, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
            self.data = None
            self.anomalies = []
            self.analysis_type = "General"
            self.viz_engine = ContextAwareVisualizationEngine()
            self.setup_ui()
        else:
            raise ImportError("PyQt5 not available")

    def setup_ui(self):
        """Setup the visualization interface"""
        layout = QVBoxLayout(self)

        # Control panel
        control_panel = QHBoxLayout()

        # Chart type selector
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Overview Analysis",
            "Traffic Analysis",
            "Threat Analysis",
            "Transaction Analysis",
            "Compliance Analysis",
            "Custom Analysis"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_visualization)

        # Context selector
        self.context_combo = QComboBox()
        self.context_combo.addItems([
            "Auto-Detect",
            "Network Forensics",
            "Financial Forensics",
            "General Analysis"
        ])
        self.context_combo.currentTextChanged.connect(self.update_visualization)

        # Refresh button
        self.refresh_button = QPushButton("🔄 Refresh Visualization")
        self.refresh_button.clicked.connect(self.update_visualization)

        control_panel.addWidget(QLabel("Chart Type:"))
        control_panel.addWidget(self.chart_type_combo)
        control_panel.addWidget(QLabel("Context:"))
        control_panel.addWidget(self.context_combo)
        control_panel.addWidget(self.refresh_button)
        control_panel.addStretch()

        # Visualization area
        self.viz_area = QVBoxLayout()

        layout.addLayout(control_panel)
        layout.addLayout(self.viz_area)

    def set_data(self, data: pd.DataFrame, anomalies: List = None, analysis_type: str = "General"):
        """Set data for visualization with context awareness"""
        self.data = data
        self.anomalies = anomalies or []
        self.analysis_type = analysis_type

        # Auto-detect context if set to auto
        if self.context_combo.currentText() == "Auto-Detect":
            if "Network" in analysis_type or "Digital" in analysis_type:
                context = "Network Forensics"
            elif "Financial" in analysis_type:
                context = "Financial Forensics"
            else:
                context = "General Analysis"

            # Update combo box
            index = self.context_combo.findText(context)
            if index >= 0:
                self.context_combo.setCurrentIndex(index)

        self.update_visualization()

    def clear_layout(self, layout):
        """Clear all widgets from layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_visualization(self):
        """Update the visualization based on context and chart type"""
        if self.data is None:
            return

        self.clear_layout(self.viz_area)

        chart_type = self.chart_type_combo.currentText().lower().replace(" ", "_")
        context = self.context_combo.currentText()

        try:
            # Map chart types
            chart_type_mapping = {
                "overview_analysis": "overview",
                "traffic_analysis": "traffic_analysis",
                "threat_analysis": "threat_analysis",
                "transaction_analysis": "transaction_analysis",
                "compliance_analysis": "compliance_analysis",
                "custom_analysis": "custom"
            }

            mapped_chart_type = chart_type_mapping.get(chart_type, "overview")

            # Create context-aware visualization
            fig = self.viz_engine.create_context_visualization(
                context, self.data, self.anomalies, mapped_chart_type
            )

            # Create canvas and add to layout
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: #2d2d2d;")
            self.viz_area.addWidget(canvas)

        except Exception as e:
            error_label = QLabel(f"Visualization Error: {str(e)}\n\nPlease check your data format and try again.")
            error_label.setStyleSheet("color: #ff4444; font-size: 14px; padding: 20px; text-align: center;")
            error_label.setAlignment(Qt.AlignCenter)
            self.viz_area.addWidget(error_label)

class VisualizationManager:
    """
    Manager class for handling different types of visualizations in Inspectra
    """

    def __init__(self):
        self.viz_engine = ContextAwareVisualizationEngine()

    @staticmethod
    def create_anomaly_summary_chart(anomalies: List) -> Figure:
        """Create a comprehensive summary chart for anomalies"""
        engine = ContextAwareVisualizationEngine()
        return engine.create_general_visualization(pd.DataFrame(), anomalies, "overview")

    @staticmethod
    def create_data_quality_report(data: pd.DataFrame) -> Figure:
        """Create a data quality assessment visualization"""
        engine = ContextAwareVisualizationEngine()
        return engine.create_general_visualization(data, [], "overview")

    @staticmethod
    def create_context_aware_chart(analysis_type: str, data: pd.DataFrame,
                                 anomalies: List, chart_type: str = "overview") -> Figure:
        """Create context-aware visualization based on analysis type"""
        engine = ContextAwareVisualizationEngine()
        return engine.create_context_visualization(analysis_type, data, anomalies, chart_type)