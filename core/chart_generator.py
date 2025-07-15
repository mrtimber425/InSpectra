# chart_generator.py
# Data visualization and charting functionality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Optional, Dict, Any, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChartGenerator:
    """Handles all data visualization and charting operations"""

    def __init__(self, figure_size: Tuple[int, int] = (10, 6), dpi: int = 100):
        self.figure_size = figure_size
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 12)

    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30,
                         title: Optional[str] = None, **kwargs) -> Figure:
        """Create histogram for a numeric column"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            # For categorical data, create bar chart
            value_counts = data.value_counts().head(20)
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                          color=self.color_palette[0], alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.set_title(title or f'Distribution of {column}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            # Numeric histogram
            n, bins_array, patches = ax.hist(data, bins=bins, alpha=0.7,
                                             color=self.color_palette[0], edgecolor='black')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.set_title(title or f'Histogram of {column}')

            # Add statistics text
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nCount: {len(data)}'
            ax.text(0.7, 0.9, stats_text, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: Optional[str] = None,
                         title: Optional[str] = None, **kwargs) -> Figure:
        """Create bar chart"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        if y_col is None:
            # Count plot
            data = df[x_col].value_counts().head(20)
            bars = ax.bar(data.index, data.values, color=self.color_palette[1], alpha=0.7)
            ax.set_xlabel(x_col)
            ax.set_ylabel('Count')
            ax.set_title(title or f'Count of {x_col}')
        else:
            # Grouped bar chart
            if pd.api.types.is_numeric_dtype(df[y_col]):
                grouped_data = df.groupby(x_col)[y_col].mean().head(20)
                bars = ax.bar(grouped_data.index, grouped_data.values,
                              color=self.color_palette[1], alpha=0.7)
                ax.set_xlabel(x_col)
                ax.set_ylabel(f'Mean {y_col}')
                ax.set_title(title or f'Mean {y_col} by {x_col}')
            else:
                # Cross-tabulation
                ct = pd.crosstab(df[x_col], df[y_col])
                ct.plot(kind='bar', ax=ax, color=self.color_palette[:len(ct.columns)])
                ax.set_xlabel(x_col)
                ax.set_ylabel('Count')
                ax.set_title(title or f'{x_col} by {y_col}')
                ax.legend(title=y_col)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom')

        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                          title: Optional[str] = None, **kwargs) -> Figure:
        """Create line chart"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        # Prepare data
        data = df[[x_col, y_col]].dropna()

        # Sort by x column if numeric or datetime
        if pd.api.types.is_numeric_dtype(data[x_col]) or pd.api.types.is_datetime64_any_dtype(data[x_col]):
            data = data.sort_values(x_col)

        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2,
                markersize=4, color=self.color_palette[2], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f'{y_col} vs {x_col}')
        ax.grid(True, alpha=0.3)

        # Format x-axis for datetime
        if pd.api.types.is_datetime64_any_dtype(data[x_col]):
            ax.tick_params(axis='x', rotation=45)
            fig.autofmt_xdate()

        # Add trend line for numeric data
        if pd.api.types.is_numeric_dtype(data[x_col]) and pd.api.types.is_numeric_dtype(data[y_col]):
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(data[x_col], p(data[x_col]), "--", alpha=0.8,
                    color='red', linewidth=2, label='Trend Line')
            ax.legend()

        fig.tight_layout()
        return fig

    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                            color_col: Optional[str] = None, size_col: Optional[str] = None,
                            title: Optional[str] = None, **kwargs) -> Figure:
        """Create scatter plot"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        data = df[[x_col, y_col]].dropna()

        if not pd.api.types.is_numeric_dtype(data[x_col]) or not pd.api.types.is_numeric_dtype(data[y_col]):
            ax.text(0.5, 0.5, 'Both columns must be numeric for scatter plot',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        # Basic scatter plot
        scatter_kwargs = {'alpha': 0.6}

        if color_col and color_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[color_col]):
                scatter = ax.scatter(data[x_col], data[y_col], c=df[color_col],
                                     cmap='viridis', **scatter_kwargs)
                fig.colorbar(scatter, ax=ax, label=color_col)
            else:
                # Categorical color
                unique_vals = df[color_col].unique()
                for i, val in enumerate(unique_vals):
                    mask = df[color_col] == val
                    ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                               label=val, color=self.color_palette[i % len(self.color_palette)],
                               **scatter_kwargs)
                ax.legend()
        else:
            ax.scatter(data[x_col], data[y_col], color=self.color_palette[3], **scatter_kwargs)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f'{y_col} vs {x_col}')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = data[x_col].corr(data[y_col])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add trend line
        if len(data) > 1:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2)

        fig.tight_layout()
        return fig

    def create_box_plot(self, df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                        title: Optional[str] = None, **kwargs) -> Figure:
        """Create box plot"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            ax.text(0.5, 0.5, 'Column must be numeric for box plot',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        if group_by and group_by in df.columns:
            # Grouped box plot
            groups = []
            labels = []
            for group in df[group_by].unique():
                if not pd.isna(group):
                    group_data = df[df[group_by] == group][column].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(str(group))

            if groups:
                box_plot = ax.boxplot(groups, labels=labels, patch_artist=True)
                # Color the boxes
                for patch, color in zip(box_plot['boxes'], self.color_palette):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xlabel(group_by)
        else:
            # Single box plot
            box_plot = ax.boxplot(data, patch_artist=True)
            box_plot['boxes'][0].set_facecolor(self.color_palette[4])
            box_plot['boxes'][0].set_alpha(0.7)

        ax.set_ylabel(column)
        ax.set_title(title or f'Box Plot of {column}')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def create_correlation_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                                   title: Optional[str] = None, **kwargs) -> Figure:
        """Create correlation heatmap"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in columns if col in df.columns and
                            pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation heatmap',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        correlation_matrix = df[numeric_cols].corr()

        # Create heatmap
        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Add labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.columns)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')

        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center",
                               color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                               fontsize=8)

        ax.set_title(title or 'Correlation Heatmap')
        fig.tight_layout()
        return fig

    def create_violin_plot(self, df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                           title: Optional[str] = None, **kwargs) -> Figure:
        """Create violin plot"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            ax.text(0.5, 0.5, 'Column must be numeric for violin plot',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        if group_by and group_by in df.columns:
            # Grouped violin plot
            clean_data = df[[column, group_by]].dropna()
            groups = clean_data[group_by].unique()
            data_list = [clean_data[clean_data[group_by] == group][column] for group in groups]

            parts = ax.violinplot(data_list, positions=range(len(groups)), showmeans=True, showmedians=True)

            # Color the violins
            for pc, color in zip(parts['bodies'], self.color_palette):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45)
            ax.set_xlabel(group_by)
        else:
            # Single violin plot
            parts = ax.violinplot([data], positions=[0], showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(self.color_palette[5])
            parts['bodies'][0].set_alpha(0.7)

        ax.set_ylabel(column)
        ax.set_title(title or f'Violin Plot of {column}')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def create_distribution_plot(self, df: pd.DataFrame, column: str,
                                 title: Optional[str] = None, **kwargs) -> Figure:
        """Create distribution plot with histogram and KDE"""
        fig = Figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(111)

        data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(data):
            ax.text(0.5, 0.5, 'Column must be numeric for distribution plot',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        # Histogram
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, density=True,
                                   color=self.color_palette[6], label='Histogram')

        # KDE
        try:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass

        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.set_title(title or f'Distribution of {column}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}'
        ax.text(0.7, 0.9, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.tight_layout()
        return fig

    def create_pair_plot(self, df: pd.DataFrame, columns: List[str],
                         hue_column: Optional[str] = None) -> Figure:
        """Create pair plot for multiple variables"""
        n_vars = len(columns)
        fig = Figure(figsize=(3 * n_vars, 3 * n_vars), dpi=self.dpi)

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ax = fig.add_subplot(n_vars, n_vars, i * n_vars + j + 1)

                if i == j:
                    # Diagonal: histogram
                    data = df[col1].dropna()
                    ax.hist(data, bins=20, alpha=0.7, color=self.color_palette[i])
                    ax.set_ylabel('Frequency')
                    if i == n_vars - 1:
                        ax.set_xlabel(col1)
                else:
                    # Off-diagonal: scatter plot
                    clean_data = df[[col1, col2]].dropna()
                    if len(clean_data) > 0:
                        if hue_column and hue_column in df.columns:
                            unique_vals = df[hue_column].unique()
                            for k, val in enumerate(unique_vals):
                                mask = df[hue_column] == val
                                ax.scatter(df.loc[mask, col2], df.loc[mask, col1],
                                           alpha=0.6, s=20,
                                           color=self.color_palette[k % len(self.color_palette)],
                                           label=val if i == 0 and j == 1 else "")
                        else:
                            ax.scatter(clean_data[col2], clean_data[col1],
                                       alpha=0.6, s=20, color=self.color_palette[7])

                if i == 0:
                    ax.set_title(col2)
                if j == 0:
                    ax.set_ylabel(col1)
                if i == n_vars - 1:
                    ax.set_xlabel(col2)

        if hue_column:
            # Add legend
            handles, labels = fig.axes[1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper right')

        fig.suptitle('Pair Plot', fontsize=16)
        fig.tight_layout()
        return fig

    def create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_cols: List[str],
                                title: Optional[str] = None, **kwargs) -> Figure:
        """Create time series plot"""
        fig = Figure(figsize=(12, 6), dpi=self.dpi)
        ax = fig.add_subplot(111)

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        # Sort by date
        df_sorted = df.sort_values(date_col)

        for i, col in enumerate(value_cols):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                clean_data = df_sorted[[date_col, col]].dropna()
                ax.plot(clean_data[date_col], clean_data[col],
                        marker='o', linewidth=2, markersize=3,
                        color=self.color_palette[i], label=col, alpha=0.8)

        ax.set_xlabel(date_col)
        ax.set_ylabel('Value')
        ax.set_title(title or 'Time Series Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        fig.autofmt_xdate()

        fig.tight_layout()
        return fig

    def create_custom_chart(self, chart_type: str, df: pd.DataFrame, **kwargs) -> Figure:
        """Create custom chart based on type"""
        chart_functions = {
            'histogram': self.create_histogram,
            'bar': self.create_bar_chart,
            'line': self.create_line_chart,
            'scatter': self.create_scatter_plot,
            'box': self.create_box_plot,
            'violin': self.create_violin_plot,
            'correlation_heatmap': self.create_correlation_heatmap,
            'distribution': self.create_distribution_plot,
            'time_series': self.create_time_series_plot
        }

        if chart_type not in chart_functions:
            fig = Figure(figsize=self.figure_size, dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Unknown chart type: {chart_type}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        return chart_functions[chart_type](df, **kwargs)

    def save_chart(self, figure: Figure, filepath: str, **kwargs):
        """Save chart to file"""
        dpi = kwargs.get('dpi', 300)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        figure.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    def get_chart_recommendations(self, df: pd.DataFrame, x_col: str,
                                  y_col: Optional[str] = None) -> List[str]:
        """Get chart type recommendations based on data types"""
        recommendations = []

        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        x_is_categorical = df[x_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[x_col])
        x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])

        if y_col:
            y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
            y_is_categorical = df[y_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[y_col])

            if x_is_numeric and y_is_numeric:
                recommendations.extend(['scatter', 'line'])
            elif x_is_categorical and y_is_numeric:
                recommendations.extend(['box', 'violin', 'bar'])
            elif x_is_datetime and y_is_numeric:
                recommendations.extend(['line', 'time_series'])
            elif x_is_categorical and y_is_categorical:
                recommendations.extend(['bar'])
        else:
            if x_is_numeric:
                recommendations.extend(['histogram', 'distribution', 'box'])
            elif x_is_categorical:
                recommendations.extend(['bar'])
            elif x_is_datetime:
                recommendations.extend(['histogram'])

        return recommendations