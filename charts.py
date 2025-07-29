"""
Data Visualization Module
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
        self.charts = {}
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot with histogram and density"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{column} - Histogram', f'{column} - Box Plot'),
            vertical_spacing=0.1
        )
        
        
        fig.add_trace(
            go.Histogram(x=df[column], name='Distribution', nbinsx=30),
            row=1, col=1
        )
        
        
        fig.add_trace(
            go.Box(y=df[column], name='Box Plot'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Distribution Analysis: {column}',
            template=self.theme,
            height=600
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            template=self.theme,
            width=600,
            height=600
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: str = None, size_col: str = None) -> go.Figure:
        """Create interactive scatter plot"""
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col, size=size_col,
            hover_data=df.columns[:5].tolist(),
            template=self.theme
        )
        
        fig.update_layout(
            title=f'Scatter Plot: {x_col} vs {y_col}',
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_cols: List[str]) -> go.Figure:
        """Create time series plot"""
        fig = go.Figure()
        
        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Time Series Analysis',
            xaxis_title=date_col,
            yaxis_title='Value',
            template=self.theme,
            hovermode='x unified'
        )
        
        return fig
    
    def create_categorical_plot(self, df: pd.DataFrame, column: str, plot_type: str = 'bar') -> go.Figure:
        """Create categorical visualization"""
        value_counts = df[column].value_counts().head(20)
        
        if plot_type == 'bar':
            fig = go.Figure(data=go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color=self.color_palette[:len(value_counts)]
            ))
            fig.update_layout(
                title=f'Category Distribution: {column}',
                xaxis_title=column,
                yaxis_title='Count'
            )
        else:  
            fig = go.Figure(data=go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=0.3
            ))
            fig.update_layout(title=f'Category Distribution: {column}')
        
        fig.update_layout(template=self.theme)
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, x_col: str = None, y_col: str = None) -> go.Figure:
        """Create box plot for outlier detection"""
        if x_col and y_col:
            fig = px.box(df, x=x_col, y=y_col, template=self.theme)
            fig.update_layout(title=f'Box Plot: {y_col} by {x_col}')
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols):
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title='Box Plots - Outlier Detection',
                template=self.theme
            )
        
        return fig
    
    def create_violin_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """Create violin plot for distribution comparison"""
        fig = px.violin(df, x=x_col, y=y_col, box=True, template=self.theme)
        fig.update_layout(title=f'Violin Plot: {y_col} by {x_col}')
        return fig
    
    def create_pca_plot(self, pca_data: np.ndarray, explained_variance: List[float]) -> go.Figure:
        """Create PCA visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PCA Components', 'Explained Variance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        
        if pca_data.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    mode='markers',
                    name='Data Points',
                    marker=dict(size=5, opacity=0.6)
                ),
                row=1, col=1
            )
        
        
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_variance))],
                y=explained_variance,
                name='Explained Variance'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Principal Component Analysis',
            template=self.theme
        )
        
        return fig
    
    def create_cluster_plot(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                           feature_cols: List[str] = None) -> go.Figure:
        """Create cluster visualization"""
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = numeric_cols[:2].tolist()
        
        if len(feature_cols) < 2:
            return go.Figure().add_annotation(text="Insufficient features for cluster plot")
        
        fig = px.scatter(
            x=df[feature_cols[0]],
            y=df[feature_cols[1]],
            color=cluster_labels.astype(str),
            title='Cluster Analysis',
            labels={'x': feature_cols[0], 'y': feature_cols[1], 'color': 'Cluster'},
            template=self.theme
        )
        
        return fig
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float]) -> go.Figure:
        """Create feature importance visualization"""
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        
        sorted_idx = np.argsort(importances)[::-1][:15]  
        
        fig = go.Figure(data=go.Bar(
            x=[importances[i] for i in sorted_idx],
            y=[features[i] for i in sorted_idx],
            orientation='h',
            marker_color=self.color_palette[:len(sorted_idx)]
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template=self.theme
        )
        
        return fig
    
    def create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        """Visualize missing data patterns"""
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Missing Data Count', 'Missing Data Percentage'),
            vertical_spacing=0.1
        )
        
        
        fig.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values, name='Count'),
            row=1, col=1
        )
        
        
        fig.add_trace(
            go.Bar(x=missing_pct.index, y=missing_pct.values, name='Percentage'),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Missing Data Analysis',
            template=self.theme,
            height=600
        )
        
        return fig
    
    def generate_all_charts(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Generate all 10 required charts"""
        charts = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        try:
            if numeric_cols:
                charts['distribution'] = self.create_distribution_plot(df, numeric_cols[0])
            if len(numeric_cols) > 1:
                charts['correlation'] = self.create_correlation_heatmap(df)
            
            if len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                charts['scatter'] = self.create_scatter_plot(
                    df, numeric_cols[0], numeric_cols[1], color_col=color_col
                )
            
            charts['boxplot'] = self.create_box_plot(df)
            
            if categorical_cols:
                charts['categorical'] = self.create_categorical_plot(df, categorical_cols[0])
            
            charts['missing_data'] = self.create_missing_data_plot(df)
            
            if 'pca_analysis' in analysis_results and 'explained_variance_ratio' in analysis_results['pca_analysis']:
                pca_info = analysis_results['pca_analysis']
                n_components = min(len(pca_info['explained_variance_ratio']), 10)
                charts['pca'] = go.Figure(data=go.Bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=pca_info['explained_variance_ratio'][:n_components]
                )).update_layout(
                    title='PCA - Explained Variance Ratio',
                    xaxis_title='Principal Components',
                    yaxis_title='Explained Variance Ratio',
                    template=self.theme
                )
            
            if 'feature_importance' in analysis_results and 'feature_importance' in analysis_results['feature_importance']:
                charts['feature_importance'] = self.create_feature_importance_plot(
                    analysis_results['feature_importance']['feature_importance']
                )
            
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if not date_cols.empty and numeric_cols:
                charts['timeseries'] = self.create_time_series_plot(
                    df, date_cols[0], numeric_cols[:3]
                )
            else:
                if len(numeric_cols) >= 2:
                    fig = go.Figure()
                    for col in numeric_cols[:3]:
                        fig.add_trace(go.Scatter(
                            y=df[col],
                            mode='lines',
                            name=col
                        ))
                    fig.update_layout(
                        title='Numeric Trends',
                        template=self.theme
                    )
                    charts['trends'] = fig
            
            if len(numeric_cols) >= 2 and categorical_cols:
                charts['violin'] = self.create_violin_plot(df, categorical_cols[0], numeric_cols[0])
            else:
                summary_stats = df.describe()
                if not summary_stats.empty:
                    fig = go.Figure(data=go.Heatmap(
                        z=summary_stats.values,
                        x=summary_stats.columns,
                        y=summary_stats.index,
                        colorscale='Viridis'
                    ))
                    fig.update_layout(
                        title='Summary Statistics Heatmap',
                        template=self.theme
                    )
                    charts['summary_stats'] = fig
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            charts['fallback'] = go.Figure().add_annotation(
                text=f"Error generating visualizations: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return charts
    
    def save_charts(self, charts: Dict[str, go.Figure], output_dir: str = "charts/"):
        """Save all charts as HTML files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, chart in charts.items():
            chart.write_html(os.path.join(output_dir, f"{name}.html"))
        
        logger.info(f"Saved {len(charts)} charts to {output_dir}")