"""
Advanced Data Analysis Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class AdvancedAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        self.insights = []
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        results = {}
        
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results['descriptive_stats'] = df[numeric_cols].describe().to_dict()
        
        # Distribution analysis of the dataset
        results['distribution_analysis'] = {}
        for col in numeric_cols:
            if len(df[col].dropna()) > 3:
                statistic, p_value = stats.shapiro(df[col].dropna()[:5000])  # Limit for performance
                results['distribution_analysis'][col] = {
                    'shapiro_test': {'statistic': statistic, 'p_value': p_value},
                    'is_normal': p_value > 0.05,
                    'skewness': stats.skew(df[col].dropna()),
                    'kurtosis': stats.kurtosis(df[col].dropna())
                }
        
        # Correlation analysis of the dataset
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            results['correlation_analysis'] = {
                'matrix': corr_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(corr_matrix)
            }
        
        # Categorical analysis of the dataset
        categorical_cols = df.select_dtypes(include=['object']).columns
        results['categorical_analysis'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            results['categorical_analysis'][col] = {
                'unique_count': len(value_counts),
                'top_categories': value_counts.head(10).to_dict(),
                'entropy': stats.entropy(value_counts.values)
            }
        
        return results
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations between variables"""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
                    })
        return strong_corr
    
    def perform_clustering_analysis(self, df: pd.DataFrame, max_clusters: int = 8) -> Dict[str, Any]:
        """Perform clustering analysis"""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2 or len(numeric_df) < 10:
            return {'error': 'Insufficient data for clustering analysis'}
        
        results = {}
        
        # Determining optimal number of clusters
        silhouette_scores = []
        inertias = []
        k_range = range(2, min(max_clusters + 1, len(numeric_df) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(numeric_df)
            silhouette_scores.append(silhouette_score(numeric_df, cluster_labels))
            inertias.append(kmeans.inertia_)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(numeric_df)
        
        results = {
            'optimal_clusters': optimal_k,
            'silhouette_score': max(silhouette_scores),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
            'feature_importance': self._calculate_cluster_feature_importance(numeric_df, cluster_labels)
        }
        
        return results
    
    def _calculate_cluster_feature_importance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        importance = {}
        for col in df.columns:
            # Calculate variance between clusters vs within clusters
            overall_var = df[col].var()
            within_cluster_var = 0
            for cluster in np.unique(labels):
                cluster_data = df[labels == cluster][col]
                if len(cluster_data) > 1:
                    within_cluster_var += cluster_data.var() * len(cluster_data)
            
            within_cluster_var /= len(df)
            between_cluster_var = overall_var - within_cluster_var
            importance[col] = between_cluster_var / overall_var if overall_var > 0 else 0
        
        return importance
    
    def perform_pca_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Insufficient numeric columns for PCA'}
        
        pca = PCA()
        pca_result = pca.fit_transform(numeric_df)
        
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumsum.tolist(),
            'n_components_95_variance': n_components_95,
            'components': pca.components_.tolist(),
            'feature_names': numeric_df.columns.tolist()
        }
        
        return results
    
    def identify_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify outliers using multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            column_data = df[col].dropna()
            if len(column_data) == 0:
                continue
                
            
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = column_data[(column_data < Q1 - 1.5 * IQR) | 
                                     (column_data > Q3 + 1.5 * IQR)]
            
            
            z_scores = np.abs(stats.zscore(column_data))
            z_outliers = column_data[z_scores > 3]
            
            outliers_info[col] = {
                'iqr_outliers_count': len(iqr_outliers),
                'z_score_outliers_count': len(z_outliers),
                'outlier_percentage': (len(iqr_outliers) / len(column_data)) * 100,
                'outlier_indices': iqr_outliers.index.tolist()
            }
        
        return outliers_info
    
    def feature_importance_analysis(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Analyze feature importance using Random Forest"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col is None or target_col not in numeric_df.columns:
            return {'error': 'Valid target column required for feature importance analysis'}
        
        X = numeric_df.drop(target_col, axis=1)
        y = numeric_df[target_col]
        
        if len(X.columns) == 0:
            return {'error': 'No features available for analysis'}
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) <= 10 and y.dtype in ['object', 'int64']
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        return {
            'feature_importance': feature_importance,
            'model_type': 'classification' if is_classification else 'regression',
            'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def generate_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            insights.append(f"Data Quality Alert: {len(high_missing)} columns have >20% missing values")
        
        
        if 'correlation_analysis' in analysis_results and 'strong_correlations' in analysis_results['correlation_analysis']:
            strong_corr = analysis_results['correlation_analysis']['strong_correlations']
            if strong_corr:
                insights.append(f"Found {len(strong_corr)} strong correlations that may indicate redundant features")
        
        
        if 'distribution_analysis' in analysis_results:
            non_normal = [col for col, info in analysis_results['distribution_analysis'].items() 
                         if not info['is_normal']]
            if non_normal:
                insights.append(f"{len(non_normal)} features show non-normal distribution - consider transformation")
        
        
        if 'clustering_analysis' in analysis_results and 'optimal_clusters' in analysis_results['clustering_analysis']:
            n_clusters = analysis_results['clustering_analysis']['optimal_clusters']
            insights.append(f"Data naturally segments into {n_clusters} distinct groups")
        
        
        if 'outliers_analysis' in analysis_results:
            high_outlier_cols = [col for col, info in analysis_results['outliers_analysis'].items() 
                               if info['outlier_percentage'] > 5]
            if high_outlier_cols:
                insights.append(f"{len(high_outlier_cols)} features have >5% outliers - may need investigation")
        
        return insights
    
    def run_comprehensive_analysis(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Run all analysis methods and compile results"""
        logger.info("Starting comprehensive data analysis...")
        
        results = {
            'statistical_analysis': self.perform_statistical_analysis(df),
            'clustering_analysis': self.perform_clustering_analysis(df),
            'pca_analysis': self.perform_pca_analysis(df),
            'outliers_analysis': self.identify_outliers(df)
        }
        
        if target_col:
            results['feature_importance'] = self.feature_importance_analysis(df, target_col)
        
        results['insights'] = self.generate_insights(df, results)
        
        logger.info("Comprehensive analysis completed")
        return results