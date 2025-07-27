"""
Configuration file for AI Analytics Agent
"""
import os
from typing import Dict, Any

class Config:
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    
    
    APP_HOST = os.getenv("APP_HOST", "localhost")
    APP_PORT = int(os.getenv("APP_PORT", "8501"))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_FILE_TYPES = ['.csv', '.xlsx', '.json']
    
    
    CHART_TYPES = [
        'histogram', 'scatter', 'line', 'bar', 'box',
        'violin', 'heatmap', 'correlation', 'distribution', 'time_series'
    ]
    
    
    ANALYSIS_PARAMS = {
        'confidence_level': 0.95,
        'significance_threshold': 0.05,
        'max_categories': 20,
        'outlier_threshold': 3.0
    }
    
    
    SYSTEM_PROMPTS = {
        'analyzer': """You are an expert data analyst. Analyze the provided data summary 
        and provide insights, patterns, and recommendations based on statistical findings.""",
        
        'decision_maker': """You are a business intelligence expert. Based on the analysis 
        provided, make strategic recommendations and identify key action items.""",
        
        'chat_assistant': """You are a helpful data analysis assistant. Answer user questions 
        about the analyzed dataset using the provided context and insights."""
    }
    
    @classmethod
    def get_chart_config(cls) -> Dict[str, Any]:
        return {
            'width': 800,
            'height': 500,
            'theme': 'plotly_white',
            'color_palette': 'viridis'
        }