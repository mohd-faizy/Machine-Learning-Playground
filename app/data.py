import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

@st.cache_data
def load_dataset(problem_type, dataset_option):
    """Load the selected dataset or a user-uploaded CSV file."""
    df, target = None, None
    
    if problem_type == "Classification":
        if dataset_option == "Iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "Breast Cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "Wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "Digits":
            data = load_digits()
            df = pd.DataFrame(data.data)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "Titanic":
            df = sns.load_dataset("titanic")
            target = 'survived'
            # Basic preprocessing for Titanic
            df = df.drop(['deck', 'embark_town', 'alive'], axis=1)
            df = df.dropna()
            # Encode categorical
            le = LabelEncoder()
            for col in ['sex', 'embarked', 'class', 'who', 'adult_male']:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
        elif dataset_option == "Penguins":
            df = sns.load_dataset("penguins")
            target = 'species'
            # Drop rows with missing values for simplicity
            df = df.dropna()
            # Encode categorical
            le = LabelEncoder()
            for col in ['island', 'sex']:
                df[col] = le.fit_transform(df[col])
        elif dataset_option == "Custom Upload":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                target = st.sidebar.selectbox("Select Target Variable", df.columns)
            else:
                return None, None
    
    elif problem_type == "Regression":
        if dataset_option == "Boston Housing":
            try:
                df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
                target = 'medv'
            except:
                st.error("Failed to load Boston Housing dataset.")
                return None, None
        elif dataset_option == "Diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "California Housing":
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
        elif dataset_option == "Tips":
            df = sns.load_dataset("tips")
            target = 'tip'
            # Encode categorical
            le = LabelEncoder()
            for col in ['sex', 'smoker', 'day', 'time']:
                df[col] = le.fit_transform(df[col])
        elif dataset_option == "Diamonds":
            df = sns.load_dataset("diamonds")
            target = 'price'
            # Encode categorical
            le = LabelEncoder()
            for col in ['cut', 'color', 'clarity']:
                df[col] = le.fit_transform(df[col])
        elif dataset_option == "Planets":
            df = sns.load_dataset("planets")
            target = 'distance'
            df = df.dropna()
            # Encode categorical
            le = LabelEncoder()
            for col in ['method']:
                df[col] = le.fit_transform(df[col])
        elif dataset_option == "Custom Upload":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                target = st.sidebar.selectbox("Select Target Variable", df.columns)
            else:
                return None, None
    
    return df, target

def preprocess_data(df, target):
    """Basic preprocessing: handle missing values and encoding."""
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].isna().sum() > 0:
            if df_processed[col].dtype in [np.float64, np.int64]:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # Encode categorical variables
    le_dict = {}
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        if col != target:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
            
    return df_processed, le_dict
