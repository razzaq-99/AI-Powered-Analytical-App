"""
Data Processing Module for AI Analytics Agent
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.cleaning_report = {}
    
    def load_data(self, file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type")
        return df

    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and generate report"""
        report = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
        }
        
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        report['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
        
        return report
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive data cleaning"""
        original_shape = df.shape
        cleaning_steps = []
        
        
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.duplicated()]
        if df.shape != original_shape:
            cleaning_steps.append(f"Removed empty rows/columns: {original_shape} -> {df.shape}")
        
        
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        if duplicates_before > 0:
            cleaning_steps.append(f"Removed {duplicates_before} duplicate rows")
        
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if df[col].isnull().sum() / len(df) > 0.5:
                    df = df.drop(col, axis=1)
                    cleaning_steps.append(f"Dropped column '{col}' (>50% missing)")
                else:
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    self.imputers[col] = imputer
                    cleaning_steps.append(f"Imputed missing values in '{col}' with median")
        
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if df[col].isnull().sum() / len(df) > 0.5:
                    df = df.drop(col, axis=1)
                    cleaning_steps.append(f"Dropped column '{col}' (>50% missing)")
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    self.imputers[col] = imputer
                    cleaning_steps.append(f"Imputed missing values in '{col}' with mode")
        
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                cleaning_steps.append(f"Capped {len(outliers)} outliers in '{col}'")
        
        self.cleaning_report = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'cleaning_steps': cleaning_steps,
            'data_reduction': f"{(1 - df.shape[0]/original_shape[0])*100:.2f}%"
        }
        
        return df, self.cleaning_report
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:  
                top_categories = df[col].value_counts().head(5).index
                for cat in top_categories:
                    df_encoded[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()
        
        if len(numeric_cols) > 0:
            df_normalized[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df_normalized
    
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict()
            },
            'statistical_summary': df.describe(include='all').to_dict(),
            'missing_data': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            summary['correlation_matrix'] = numeric_df.corr().to_dict()
        
        return summary