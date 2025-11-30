import streamlit as st
from app.ui import setup_page, render_header, render_sidebar_header, render_footer
from app.data import load_dataset, preprocess_data
from app.analysis import render_data_overview, render_analysis, render_model_evaluation
from app.models import train_models, get_models
from app.features import render_feature_engineering
import pandas as pd

# Setup Page
setup_page()

# Header
render_header()

# Sidebar
render_sidebar_header()

# Sidebar Configuration
problem_type = st.sidebar.selectbox(
    "📊 Select Problem Type", 
    ["Classification", "Regression"]
)

# Dataset Selection
st.sidebar.markdown('<p class="sidebar-header">Dataset Selection</p>', unsafe_allow_html=True)
if problem_type == "Classification":
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris", "Titanic", "Penguins", "Breast Cancer", "Wine", "Digits", "Custom Upload"]
    )
else:
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["Boston Housing", "Diabetes", "California Housing", "Tips", "Diamonds", "Planets", "Custom Upload"]
    )

# Advanced Settings
with st.sidebar.expander("⚙️ Data & Model Settings", expanded=False):
    test_size = st.slider("Test Set Proportion", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random State", value=42, step=1)
    scaler_option = st.selectbox("Scaler (for Training)", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

render_footer()

# Load Data
df, target = load_dataset(problem_type, dataset_option)

if df is not None:
    # Initial Preprocessing (Missing values, Encoding)
    df_processed, _ = preprocess_data(df, target)
    
    # Initialize or update session state for engineered data
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = dataset_option
        st.session_state.df_engineered = df_processed.copy()
        
    if st.session_state.current_dataset != dataset_option:
        st.session_state.df_engineered = df_processed.copy()
        st.session_state.current_dataset = dataset_option
        
    if 'df_engineered' not in st.session_state:
        st.session_state.df_engineered = df_processed.copy()
        
    # Tabs
    tabs = st.tabs(["📊 Data Overview", "🛠️ Feature Engineering", "🔍 Data Analysis", "🤖 Model Training", "🔮 Predictions"])
    
    # Tab 1: Overview
    with tabs[0]:
        render_data_overview(st.session_state.df_engineered, target)
        
    # Tab 2: Feature Engineering
    with tabs[1]:
        # Update session state with engineered data
        st.session_state.df_engineered = render_feature_engineering(st.session_state.df_engineered, target, problem_type)
        
    # Tab 3: Analysis
    with tabs[2]:
        render_analysis(st.session_state.df_engineered, target, problem_type)
        
    # Tab 4: Training
    with tabs[3]:
        st.markdown('<p class="subheader">Model Training & Evaluation</p>', unsafe_allow_html=True)
        
        available_models = get_models(problem_type)
        selected_models = st.multiselect(
            "Select Models to Train",
            list(available_models.keys()),
            default=list(available_models.keys())[:2]
        )
        
        if st.button("🚀 Train Selected Models"):
            if not selected_models:
                st.warning("Please select at least one model.")
            else:
                results_df, predictions, probs, trained_models, y_test = train_models(
                    st.session_state.df_engineered, target, selected_models, problem_type, 
                    test_size, random_state, scaler_option
                )
                
                if results_df is not None:
                    st.markdown("### Performance Metrics")
                    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
                    
                    # Plot results
                    import plotly.express as px
                    metric_to_plot = "Accuracy" if problem_type == "Classification" else "R2 Score"
                    fig_res = px.bar(results_df.reset_index(), x="Model", y=metric_to_plot, 
                                     color="Model", title=f"Model Comparison - {metric_to_plot}",
                                     template="plotly_dark")
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                    # Advanced Evaluation
                    render_model_evaluation(y_test, predictions, probs, problem_type)

    # Tab 5: Predictions (Placeholder for now)
    with tabs[4]:
        st.info("Prediction interface coming soon! Use the Model Training tab to evaluate models.")

else:
    st.info("Please upload a CSV file or select a dataset to get started.")
