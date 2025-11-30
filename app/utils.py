import streamlit as st
import pandas as pd
import numpy as np

def format_metrics(metrics_dict):
    """Format metrics dictionary for display."""
    return {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_dict.items()}

def get_column_types(df):
    """Get column types separated by numeric and categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols
