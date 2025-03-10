import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os      # Added for directory handling
import joblib  # For saving and loading trained models
import time    # For progress indicators

# Sklearn datasets for loading sample data
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing

# Sklearn utilities for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# -------------------------------
# Custom CSS and Page Configuration
# -------------------------------
st.set_page_config(page_title="ML Playground", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Base styles for the entire app */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Fix for empty containers */
    .element-container {
        background-color: transparent;
    }
    
    /* Empty div fix */
    div:empty {
        display: none !important;
    }
    
    /* Fix for white bars/containers */
    .block-container, .css-1y4p8pa, .css-1r6slb0, .css-12oz5g7 {
        background-color: #1E1E1E;
    }
    
    /* Card styling with dark theme */
    .card {
        padding: 1.8rem;
        border-radius: 0.8rem;
        background-color: #2D2D2D;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #3D3D3D;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #FFFFFF;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Other styles with dark theme */
    .main-header {
        font-size: 2.8rem;
        color: #BB86FC;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 3px solid #7B1FA2;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .subheader {
        font-size: 1.8rem;
        color: #BB86FC;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
        border-left: 4px solid #7B1FA2;
        padding-left: 10px;
    }
    
    .metric-card {
        background-color: #2D2D2D;
        border-radius: 0.8rem;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        border: 1px solid #3D3D3D;
        color: #FFFFFF;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #BB86FC;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #BB86FC;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-radius: 8px;
        overflow: hidden;
        background-color: #2D2D2D;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #3D3D3D;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 12px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #7B1FA2;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(123, 31, 162, 0.3);
    }
    
    /* Input styling */
    .prediction-input {
        background-color: #2D2D2D;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid #3D3D3D;
        transition: all 0.3s ease;
        color: #FFFFFF;
    }
    
    .prediction-input:hover {
        background-color: #3D3D3D;
        border-color: #BB86FC;
    }
    
    .prediction-input label {
        font-weight: 500;
        color: #BB86FC;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #BB86FC;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #7B1FA2;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #7B1FA2;
        color: white !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #6A1B9A;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
        color: white !important;
    }
    
    .stButton > button:focus:not(:active) {
        border-color: #BB86FC;
        color: white !important;
        background-color: #7B1FA2;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Fix for all text elements */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    /* Fix for button text */
    .stButton > button span {
        color: white !important;
    }
    
    /* Fix for expander text */
    .streamlit-expanderHeader {
        color: #FFFFFF;
        font-weight: 500;
    }
    
    /* Fix for form elements */
    .stRadio > div, .stMultiSelect > div, .stSlider > div, 
    .stCheckbox > label, .stSelectbox > div > div {
        color: #FFFFFF;
    }
    
    /* Fix for number inputs */
    .stNumberInput > div > div > input {
        color: #FFFFFF;
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    /* Fix for select boxes */
    .stSelectbox > div > div {
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    /* Fix for multiselect */
    .stMultiSelect > div {
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    /* Fix for tables */
    .dataframe {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    
    .dataframe th {
        background-color: #3D3D3D;
        color: #FFFFFF;
    }
    
    .dataframe td {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Main header with custom styling
st.markdown('<h1 class="main-header">🧪 Machine Learning Playground</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #555;">
        Explore, train, and evaluate machine learning models with an intuitive interface.
        Select options from the sidebar!
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.image("assets/ml-playground.png", width=300)
    st.markdown('<p class="sidebar-header">ML Playground Configuration</p>', unsafe_allow_html=True)
    
    # Problem type selection with icons
    problem_type = st.selectbox(
        "📊 Select Problem Type", 
        ["Classification", "Regression", "Custom Upload"]
    )

    # Dataset selection based on problem type
    st.markdown('<p class="sidebar-header">Dataset Selection</p>', unsafe_allow_html=True)
    if problem_type in ["Classification", "Regression"]:
        if problem_type == "Classification":
            dataset_option = st.selectbox(
                "Select Dataset",
                ["Iris", "Titanic", "Breast Cancer", "Wine", "Digits", "Custom Upload"]
            )
        else:
            dataset_option = st.selectbox(
                "Select Dataset",
                ["Boston Housing", "Diabetes", "California Housing", "Custom Upload"]
            )
    else:
        dataset_option = "Custom Upload"

    # Collapsible sections for advanced settings
    with st.expander("⚙️ Data & Model Settings", expanded=False):
        test_size = st.slider("Test Set Proportion", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42, step=1)
        scaler_option = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])

    st.sidebar.header("About")
    st.sidebar.caption("ML PlayGround | by faizy")

# -------------------------------
# Data Loading Function
# -------------------------------
@st.cache_data  # Cache the function to improve performance
def load_dataset(problem_type, dataset_option):
    """Load the selected dataset or a user-uploaded CSV file."""
    # ... existing code ...
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
            df = df.dropna(subset=[target])  # Drop rows with missing target
        elif dataset_option == "Custom Upload":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                target = st.sidebar.selectbox("Select Target Variable", df.columns)
            else:
                return None, None
        else:
            df, target = None, None
    elif problem_type == "Regression":
        if dataset_option == "Boston Housing":
            df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
            target = 'medv'
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
        elif dataset_option == "Custom Upload":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                target = st.sidebar.selectbox("Select Target Variable", df.columns)
            else:
                return None, None
        else:
            df, target = None, None
    else:  # Custom Upload
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target = st.sidebar.selectbox("Select Target Variable", df.columns)
        else:
            return None, None
    return df, target

# Load the dataset
df, target = load_dataset(problem_type, dataset_option)

# -------------------------------
# Main Content
# -------------------------------
if df is not None:
    # Create tabs for better organization
    tabs = st.tabs(["📊 Data Overview", "🔍 Data Analysis", "🤖 Model Training", "📈 Visualizations", "🔮 Predictions"])
    
    # -------------------------------
    # Tab 1: Data Overview
    # -------------------------------
    with tabs[0]:
        st.markdown('<p class="subheader">Dataset Overview</p>', unsafe_allow_html=True)
        
        # Basic dataset info
        st.write(f"**Dataset**: {dataset_option}")
        st.write(f"**Problem Type**: {problem_type}")
        st.write(f"**Rows**: {df.shape[0]}, **Columns**: {df.shape[1]}")
        st.write(f"**Target Variable**: {target}")
        
        # Import plotly for better tables
        import plotly.graph_objects as go
        
        # Data preview with Plotly
        st.subheader("Data Preview (First 5 Rows)")
        preview_df = df.head(5).reset_index()
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Index'] + list(df.columns),
                fill_color='#7B1FA2',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[preview_df['index']] + [preview_df[col] for col in df.columns],
                fill_color='#2D2D2D',
                align='left',
                font=dict(color='white', size=11)
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Descriptive statistics with Plotly
        st.subheader("Descriptive Statistics")
        stats_df = df.describe().reset_index()
        
        fig_stats = go.Figure(data=[go.Table(
            header=dict(
                values=['Statistic'] + list(df.columns),
                fill_color='#7B1FA2',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[stats_df['index']] + [stats_df[col].round(2) for col in stats_df.columns],
                fill_color='#2D2D2D',
                align='left',
                font=dict(color='white', size=11)
            )
        )])
        
        fig_stats.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        st.plotly_chart(fig_stats, use_container_width=True)
        
        # Column information with Plotly
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null Count': df.notna().sum(),
            'Unique Values': df.nunique()
        })
        
        fig_col = go.Figure(data=[go.Table(
            header=dict(
                values=list(col_info.columns),
                fill_color='#7B1FA2',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[col_info[col] for col in col_info.columns],
                fill_color='#2D2D2D',
                align='left',
                font=dict(color='white', size=11)
            )
        )])
        
        fig_col.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        st.plotly_chart(fig_col, use_container_width=True)
        
        # Missing values with Plotly
        st.subheader("Missing Values")
        missing = df.isna().sum()
        if missing.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing.index[missing > 0],
                'Missing Count': missing[missing > 0],
                'Percentage': [(count/len(df))*100 for count in missing[missing > 0]]
            })
            missing_df['Percentage'] = missing_df['Percentage'].round(1).astype(str) + '%'
            
            fig_missing = go.Figure(data=[go.Table(
                header=dict(
                    values=list(missing_df.columns),
                    fill_color='#7B1FA2',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[missing_df[col] for col in missing_df.columns],
                    fill_color='#2D2D2D',
                    align='left',
                    font=dict(color='white', size=11)
                )
            )])
            
            fig_missing.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=200
            )
            
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values found in the dataset.")

    # -------------------------------
    # Tab 2: Data Analysis
    # -------------------------------
    with tabs[1]:
        st.markdown('<p class="subheader">Data Analysis</p>', unsafe_allow_html=True)
        
        # Handle missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].median()
                else:
                    df[col] = df[col].mode()[0]

        # Encode categorical variables
        for col in df.select_dtypes(include=['object', 'category']).columns:
            try:
                df[col] = LabelEncoder().fit_transform(df[col])
            except Exception as e:
                st.write(f"Error encoding {col}: {e}")

        # Distribution of target variable
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Target Variable Distribution**")
        fig_target, ax_target = plt.subplots(figsize=(10, 6))
        if problem_type == "Classification":
            target_counts = df[target].value_counts().sort_index()
            sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax_target)
            ax_target.set_title(f"Distribution of {target}")
            ax_target.set_xlabel(target)
            ax_target.set_ylabel("Count")
        else:
            sns.histplot(df[target], kde=True, ax=ax_target)
            ax_target.set_title(f"Distribution of {target}")
            ax_target.set_xlabel(target)
            ax_target.set_ylabel("Frequency")
        st.pyplot(fig_target)
        st.markdown('</div>', unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature distributions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Feature Distributions**")
        selected_features_dist = st.multiselect(
            "Select features to visualize distributions",
            options=[col for col in df.columns if col != target],
            default=[col for col in df.columns if col != target][:3],
            key="feature_dist_multiselect" # Unique key
        )
        
        if selected_features_dist:
            fig_dist = plt.figure(figsize=(12, 4 * len(selected_features_dist)))
            for i, feature in enumerate(selected_features_dist):
                ax = fig_dist.add_subplot(len(selected_features_dist), 1, i+1)
                sns.histplot(df[feature], kde=True, ax=ax)
                ax.set_title(f"Distribution of {feature}")
            fig_dist.tight_layout()
            st.pyplot(fig_dist)
        st.markdown('</div>', unsafe_allow_html=True)

        # Box plots for features
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Feature Box Plots**")
        selected_features_box = st.multiselect(
            "Select features to visualize box plots",
            options=[col for col in df.columns if col != target],
            default=[col for col in df.columns if col != target][:3],
            key="feature_box_multiselect" # Unique key
        )

        if selected_features_box:
            fig_box = plt.figure(figsize=(12, 4 * len(selected_features_box)))
            for i, feature in enumerate(selected_features_box):
                ax = fig_box.add_subplot(len(selected_features_box), 1, i+1)
                if problem_type == "Classification":
                    sns.boxplot(x=target, y=feature, data=df, ax=ax)
                else:
                    sns.boxplot(y=feature, data=df, ax=ax) # Only y for regression
                ax.set_title(f"Box Plot of {feature}")
            fig_box.tight_layout()
            st.pyplot(fig_box)
        st.markdown('</div>', unsafe_allow_html=True)

        # Scatter plots for feature relationships
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Feature Scatter Plots**")
        feature_x = st.selectbox("Select feature for X-axis (Scatter Plot)",
                                 options=[col for col in df.columns if col != target],
                                 key="scatter_x_selectbox")
        feature_y = st.selectbox("Select feature for Y-axis (Scatter Plot)",
                                 options=[col for col in df.columns if col != target and col != feature_x],
                                 key="scatter_y_selectbox")

        if feature_x and feature_y:
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            if problem_type == "Classification":
                sns.scatterplot(x=feature_x, y=feature_y, hue=target, data=df, ax=ax_scatter)
            else:
                sns.scatterplot(x=feature_x, y=feature_y, data=df, ax=ax_scatter)
            ax_scatter.set_title(f"Scatter Plot: {feature_x} vs {feature_y}")
            st.pyplot(fig_scatter)
        st.markdown('</div>', unsafe_allow_html=True)

        # Pair plots for feature relationships (Classification only)
        if problem_type == "Classification":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Pair Plot of Features**")
            pair_plot_features = st.multiselect(
                "Select features for Pair Plot (max 5 for performance)",
                options=[col for col in df.columns if col != target],
                default=[col for col in df.columns if col != target][:min(3, len([col for col in df.columns if col != target]))],
                key="pair_plot_features_multiselect"
            )
            if pair_plot_features:
                fig_pair = sns.pairplot(df[pair_plot_features + [target]], hue=target, corner=True)
                st.pyplot(fig_pair)
            st.markdown('</div>', unsafe_allow_html=True)

        # Violin plots for feature distribution by target (Classification) or feature itself (Regression)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Violin Plots**")
        selected_features_violin = st.multiselect(
            "Select features for Violin Plots",
            options=[col for col in df.columns if col != target],
            default=[col for col in df.columns if col != target][:min(2, len([col for col in df.columns if col != target]))],
            key="violin_plot_features_multiselect"
        )

        if selected_features_violin:
            fig_violin = plt.figure(figsize=(12, 4 * len(selected_features_violin)))
            for i, feature in enumerate(selected_features_violin):
                ax = fig_violin.add_subplot(len(selected_features_violin), 1, i+1)
                if problem_type == "Classification":
                    sns.violinplot(x=target, y=feature, data=df, ax=ax)
                else: # Regression - just show distribution of feature
                    sns.violinplot(y=feature, data=df, ax=ax)
                ax.set_title(f"Violin Plot of {feature}")
            fig_violin.tight_layout()
            st.pyplot(fig_violin)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Tab 3: Model Training
    # -------------------------------
    with tabs[2]:
        st.markdown('<p class="subheader">Model Training & Evaluation</p>', unsafe_allow_html=True)
        
        # Data Splitting & Scaling
        X = df.drop(target, axis=1)  # Features
        y = df[target]  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)

        # Model Selection
        if problem_type == "Classification":
            model_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC(probability=True),
                "k-Nearest Neighbors": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Neural Network": MLPClassifier(max_iter=1000),
                "Naive Bayes": GaussianNB()
            }
        else:
            model_dict = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Regressor": SVR(),
                "k-Nearest Neighbors": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Neural Network": MLPRegressor(max_iter=1000)
            }

        # Let user select models with a more visual interface
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Select Models to Train")
        selected_models = []

        # Create a more visual model selection interface
        model_cols = st.columns(2)
        for i, (model_name, model) in enumerate(model_dict.items()):
            with model_cols[i % 2]:
                model_selected = st.checkbox(
                    model_name, 
                    value=(i < 2),
                    key=f"model_{model_name}"
                )
                if model_selected:
                    selected_models.append(model_name)
                    
                # Add a short description for each model
                model_descriptions = {
                    "Logistic Regression": "Simple and interpretable linear model for classification",
                    "Linear Regression": "Simple and interpretable linear model for regression",
                    "Decision Tree": "Tree-based model with good interpretability",
                    "Random Forest": "Ensemble of trees with high accuracy",
                    "Support Vector Machine": "Powerful for complex decision boundaries",
                    "Support Vector Regressor": "Powerful for complex regression tasks",
                    "k-Nearest Neighbors": "Instance-based learning using proximity",
                    "Gradient Boosting": "High-performance ensemble method",
                    "Neural Network": "Deep learning model for complex patterns",
                    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem"
                }
                
                if model_name in model_descriptions:
                    st.markdown(f"<small style='color: #666;'>{model_descriptions[model_name]}</small>", 
                               unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not selected_models:
            st.warning("Please select at least one model to train.")
            st.stop()

        # Train button with progress indicator
        st.markdown('<div class="card">', unsafe_allow_html=True)
        train_button = st.button("🚀 Train Selected Models")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Initialize session state variables if they don't exist
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None
        if 'model_predictions' not in st.session_state:
            st.session_state.model_predictions = {}
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = []
        
        if train_button:
            results = []
            model_predictions = {}  # Store predictions for visualization
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                model = model_dict[model_name]
                
                # Add a small delay to show progress
                time.sleep(0.5)
                
                model.fit(X_train_scaled, y_train)  # Train the model
                y_pred = model.predict(X_test_scaled)  # Make predictions
                model_predictions[model_name] = y_pred

                # Save the trained model to the 'saved_models' folder
                model_path = os.path.join('saved_models', f'{model_name.replace(" ", "_")}.pkl')
                joblib.dump(model, model_path)

                # Evaluate the model
                if problem_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted')
                    rec = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results.append({
                        "Model": model_name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1
                    })
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results.append({
                        "Model": model_name,
                        "MSE": mse,
                        "R2 Score": r2
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("Training complete!")
            
            # Store results in session state
            if results:
                results_df = pd.DataFrame(results).set_index("Model")
                st.session_state.results_df = results_df
                st.session_state.model_predictions = model_predictions
                st.session_state.trained_models = selected_models
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("### Performance Metrics")
                
                # Create a custom styled table for better visibility in dark mode
                html_table = '<table style="width:100%; border-collapse: collapse; margin-top: 10px;">'
                
                # Table header
                html_table += '<thead><tr style="background-color: #3D3D3D; color: white;">'
                html_table += '<th style="padding: 12px; text-align: left; border-bottom: 2px solid #BB86FC;">Model</th>'
                
                # Get column names excluding the index
                for col in results_df.columns:
                    html_table += f'<th style="padding: 12px; text-align: center; border-bottom: 2px solid #BB86FC;">{col}</th>'
                
                html_table += '</tr></thead><tbody>'
                
                # Table rows
                for model_name in results_df.index:
                    html_table += f'<tr style="background-color: #2D2D2D; border-bottom: 1px solid #3D3D3D;">'
                    html_table += f'<td style="padding: 10px; text-align: left; color: #BB86FC; font-weight: 500;">{model_name}</td>'
                    
                    for col in results_df.columns:
                        value = results_df.loc[model_name, col]
                        html_table += f'<td style="padding: 10px; text-align: center; color: white;">{value:.4f}</td>'
                    
                    html_table += '</tr>'
                
                html_table += '</tbody></table>'
                
                st.markdown(html_table, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metrics in a more visual way
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("### Key Metrics Visualization")
                
                if problem_type == "Classification":
                    metric_cols = st.columns(len(selected_models))
                    for i, model_name in enumerate(selected_models):
                        with metric_cols[i]:
                            model_result = results_df.loc[model_name]
                            st.markdown(f"""
                            <div class="metric-card">
                                <div>{model_name}</div>
                                <div class="metric-value">{model_result['Accuracy']:.3f}</div>
                                <div class="metric-label">Accuracy</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    metric_cols = st.columns(len(selected_models))
                    for i, model_name in enumerate(selected_models):
                        with metric_cols[i]:
                            model_result = results_df.loc[model_name]
                            st.markdown(f"""
                            <div class="metric-card">
                                <div>{model_name}</div>
                                <div class="metric-value">{model_result['R2 Score']:.3f}</div>
                                <div class="metric-label">R² Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bar chart of model performance
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("### Model Comparison")
                
                if problem_type == "Classification":
                    fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                    results_df['Accuracy'].sort_values().plot(kind='barh', ax=ax_perf, color='#7B1FA2')
                    ax_perf.set_title("Model Accuracy Comparison")
                    ax_perf.set_xlabel("Accuracy")
                    st.pyplot(fig_perf)
                else:
                    fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                    results_df['R2 Score'].sort_values().plot(kind='barh', ax=ax_perf, color='#7B1FA2')
                    ax_perf.set_title("Model R² Score Comparison")
                    ax_perf.set_xlabel("R² Score")
                    st.pyplot(fig_perf)
                st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Tab 4: Visualizations
    # -------------------------------
    with tabs[3]:
        st.markdown('<p class="subheader">Model Visualizations</p>', unsafe_allow_html=True)
        
        if st.session_state.results_df is not None:
            # Use the trained models for visualization
            if st.session_state.trained_models:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                viz_col1, viz_col2 = st.columns([1, 3])
                
                with viz_col1:
                    viz_model = st.selectbox("Select Model", st.session_state.trained_models)
                    
                    # Filter visualization options based on problem type
                    if problem_type == "Classification":
                        viz_options = ["Confusion Matrix", "ROC Curve", "Feature Importance", "Accuracy Score", "Precision Score", "Recall Score", "F1 Score"]
                    else:  # Regression
                        viz_options = ["Actual vs Predicted", "Residual Plot", "Feature Importance", "MSE", "R2 Score"]
                    
                    viz_option = st.radio("Select Visualization", viz_options)
                
                with viz_col2:
                    try:
                        # Load the model from saved file
                        model_path = os.path.join('saved_models', f'{viz_model.replace(" ", "_")}.pkl')
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                        else:
                            model = model_dict[viz_model]  # Fallback to the model dictionary
                        
                        if viz_option == "Confusion Matrix" and problem_type == "Classification":
                            if viz_model in st.session_state.model_predictions:
                                y_pred = st.session_state.model_predictions[viz_model]
                                cm = confusion_matrix(y_test, y_pred)
                                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_title(f"Confusion Matrix - {viz_model}")
                                ax_cm.set_xlabel("Predicted Label")
                                ax_cm.set_ylabel("True Label")
                                st.pyplot(fig_cm)
                            else:
                                st.warning(f"No predictions available for {viz_model}. Please retrain the model.")
                        
                        elif viz_option == "ROC Curve" and problem_type == "Classification":
                            if len(np.unique(y_test)) == 2:  # Binary classification check
                                if viz_model in st.session_state.model_predictions:
                                    if hasattr(model, "predict_proba"):
                                        y_prob = model.predict_proba(X_test_scaled)[:, 1]
                                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                                        roc_auc = auc(fpr, tpr)
                                        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                                        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='#7B1FA2', linewidth=2)
                                        ax_roc.plot([0, 1], [0, 1], 'k--')
                                        ax_roc.set_xlabel("False Positive Rate")
                                        ax_roc.set_ylabel("True Positive Rate")
                                        ax_roc.set_title(f"ROC Curve - {viz_model}")
                                        ax_roc.legend()
                                        st.pyplot(fig_roc)
                                    elif hasattr(model, "decision_function"):
                                        y_score = model.decision_function(X_test_scaled)
                                        fpr, tpr, _ = roc_curve(y_test, y_score)
                                        roc_auc = auc(fpr, tpr)
                                        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                                        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='#7B1FA2', linewidth=2)
                                        ax_roc.plot([0, 1], [0, 1], 'k--')
                                        ax_roc.set_xlabel("False Positive Rate")
                                        ax_roc.set_ylabel("True Positive Rate")
                                        ax_roc.set_title(f"ROC Curve - {viz_model}")
                                        ax_roc.legend()
                                        st.pyplot(fig_roc)
                                    else:
                                        st.info(f"ROC Curve is not available for {viz_model} as it doesn't support probability predictions.")
                                else:
                                    st.warning(f"No predictions available for {viz_model}. Please retrain the model.")
                            else:
                                st.info("ROC Curve is available only for binary classification problems. This dataset has more than two classes.")
                        elif viz_option == "Accuracy Score" and problem_type == "Classification":
                            if viz_model in st.session_state.results_df.index:
                                accuracy = st.session_state.results_df.loc[viz_model, 'Accuracy']
                                st.metric(label=f"{viz_model} Accuracy", value=f"{accuracy:.4f}")
                            else:
                                st.warning(f"Accuracy score not available for {viz_model}. Please train the model.")
                        elif viz_option == "Precision Score" and problem_type == "Classification":
                            if viz_model in st.session_state.results_df.index:
                                precision = st.session_state.results_df.loc[viz_model, 'Precision']
                                st.metric(label=f"{viz_model} Precision", value=f"{precision:.4f}")
                            else:
                                st.warning(f"Precision score not available for {viz_model}. Please train the model.")
                        elif viz_option == "Recall Score" and problem_type == "Classification":
                            if viz_model in st.session_state.results_df.index:
                                recall = st.session_state.results_df.loc[viz_model, 'Recall']
                                st.metric(label=f"{viz_model} Recall", value=f"{recall:.4f}")
                            else:
                                st.warning(f"Recall score not available for {viz_model}. Please train the model.")
                        elif viz_option == "F1 Score" and problem_type == "Classification":
                            if viz_model in st.session_state.results_df.index:
                                f1 = st.session_state.results_df.loc[viz_model, 'F1 Score']
                                st.metric(label=f"{viz_model} F1 Score", value=f"{f1:.4f}")
                            else:
                                st.warning(f"F1 Score not available for {viz_model}. Please train the model.")
                        elif viz_option == "MSE" and problem_type == "Regression":
                            if viz_model in st.session_state.results_df.index:
                                mse_score = st.session_state.results_df.loc[viz_model, 'MSE']
                                st.metric(label=f"{viz_model} MSE", value=f"{mse_score:.4f}")
                            else:
                                st.warning(f"MSE not available for {viz_model}. Please train the model.")
                        elif viz_option == "R2 Score" and problem_type == "Regression":
                            if viz_model in st.session_state.results_df.index:
                                r2_score_val = st.session_state.results_df.loc[viz_model, 'R2 Score']
                                st.metric(label=f"{viz_model} R² Score", value=f"{r2_score_val:.4f}")
                            else:
                                st.warning(f"R2 Score not available for {viz_model}. Please train the model.")
                        elif viz_option == "Actual vs Predicted" and problem_type == "Regression":
                            if viz_model in st.session_state.model_predictions:
                                y_pred = st.session_state.model_predictions[viz_model]
                                fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
                                ax_pred.scatter(y_test, y_pred, alpha=0.5, color='#7B1FA2')
                                ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                                ax_pred.set_xlabel("Actual")
                                ax_pred.set_ylabel("Predicted")
                                ax_pred.set_title(f"Actual vs Predicted - {viz_model}")
                                st.pyplot(fig_pred)
                            else:
                                st.warning(f"No predictions available for {viz_model}. Please retrain the model.")
                        
                        elif viz_option == "Residual Plot" and problem_type == "Regression":
                            if viz_model in st.session_state.model_predictions:
                                y_pred = st.session_state.model_predictions[viz_model]
                                residuals = y_test - y_pred
                                fig_res, ax_res = plt.subplots(figsize=(8, 6))
                                ax_res.scatter(y_pred, residuals, alpha=0.5, color='#7B1FA2')
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title(f"Residual Plot - {viz_model}")
                                st.pyplot(fig_res)
                            else:
                                st.warning(f"No predictions available for {viz_model}. Please retrain the model.")
                        
                        elif viz_option == "Feature Importance":
                            if hasattr(model, "feature_importances_"):
                                importances = model.feature_importances_
                                feat_imp = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
                                sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15), ax=ax_fi, palette='viridis')
                                ax_fi.set_title(f"Feature Importance - {viz_model}")
                                st.pyplot(fig_fi)
                            elif hasattr(model, "coef_"):
                                # For linear models
                                if len(model.coef_.shape) == 1:
                                    # For regression or binary classification
                                    importances = np.abs(model.coef_)
                                    feat_imp = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)
                                else:
                                    # For multi-class classification
                                    importances = np.mean(np.abs(model.coef_), axis=0)
                                    feat_imp = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)
                                
                                fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
                                sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15), ax=ax_fi, palette='viridis')
                                ax_fi.set_title(f"Feature Importance (Coefficient Magnitude) - {viz_model}")
                                st.pyplot(fig_fi)
                            else:
                                st.info(f"Feature importance is not available for {viz_model}. This model type doesn't provide feature importance scores.")
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
                        st.info("Try retraining the model or selecting a different visualization option.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Train models in the 'Model Training' tab to view visualizations.")

    # -------------------------------
    # Tab 5: Predictions
    # -------------------------------
    with tabs[4]:
        st.markdown('<p class="subheader">Make Predictions</p>', unsafe_allow_html=True)
        
        if st.session_state.results_df is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### Input Features")
            
            # Add feature description or statistics
            with st.expander("Feature Information", expanded=False):
                feature_info = pd.DataFrame({
                    'Feature': X.columns,
                    'Min': X.min(),
                    'Max': X.max(),
                    'Mean': X.mean(),
                    'Median': X.median()
                })
                st.dataframe(feature_info)
            
            # More organized input interface with columns
            num_features = len(X.columns)
            num_cols = 3  # Number of columns for layout
            num_rows = (num_features + num_cols - 1) // num_cols  # Ceiling division
            
            input_data = {}
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col in range(num_cols):
                    idx = row * num_cols + col
                    if idx < num_features:
                        feature = X.columns[idx]
                        with cols[col]:
                            st.markdown(f"<div class='prediction-input'>", unsafe_allow_html=True)
                            # Add feature name with better styling
                            st.markdown(f"<label style='font-weight: 500; color: #BB86FC;'>{feature}</label>", 
                                       unsafe_allow_html=True)
                            # Add min/max as a hint
                            min_val = float(X[feature].min())
                            max_val = float(X[feature].max())
                            med_val = float(X[feature].median())
                            
                            input_data[feature] = st.number_input(
                                "",  # Empty label since we added it above
                                min_value=min_val,
                                max_value=max_val,
                                value=med_val,
                                format="%.2f",
                                help=f"Range: {min_val:.2f} to {max_val:.2f}, Median: {med_val:.2f}",
                                key=f"feature_input_{feature}"
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
            
            predict_col1, predict_col2 = st.columns([1, 3])
            with predict_col1:
                predict_button = st.button("🔮 Predict", use_container_width=True)
            
            if predict_button:
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                
                with predict_col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("### Prediction Results")
                    
                    # Create a more visual results display
                    for model_name in st.session_state.trained_models:
                        model = joblib.load(os.path.join('saved_models', f'{model_name.replace(" ", "_")}.pkl'))
                        prediction = model.predict(input_scaled)
                        
                        # Add confidence for classification if available
                        confidence = None
                        if problem_type == "Classification" and hasattr(model, "predict_proba"):
                            proba = model.predict_proba(input_scaled)[0]
                            confidence = proba[int(prediction[0])] if len(proba) > int(prediction[0]) else None
                        
                        # Display with better styling
                        st.markdown(f"""
                        <div style="background-color: #2D2D2D; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 600; color: #BB86FC;">{model_name}</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #7B1FA2;">
                                    {prediction[0] if problem_type == "Classification" else f"{prediction[0]:.4f}"}
                                </div>
                            </div>
                            {f'<div style="text-align: right; font-size: 0.8rem; color: #666;">Confidence: {confidence:.2%}</div>' if confidence else ''}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please train models in the 'Model Training' tab first to make predictions.")