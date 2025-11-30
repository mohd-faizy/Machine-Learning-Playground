import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def render_data_overview(df, target):
    """Render detailed data overview tab."""
    st.markdown('<p class="subheader">Dataset Overview</p>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        missing_count = df.isna().sum().sum()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{missing_count}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
    with col4:
        duplicates = df.duplicated().sum()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{duplicates}</div><div class="metric-label">Duplicates</div></div>', unsafe_allow_html=True)
    
    # Data Preview
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Detailed Statistics
    st.markdown("### 📊 Detailed Statistics")
    
    # Numeric Stats
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats_df = numeric_df.describe().T
        stats_df['skew'] = numeric_df.skew()
        stats_df['kurtosis'] = numeric_df.kurtosis()
        stats_df = stats_df.reset_index().rename(columns={'index': 'Feature'})
        
        # Rounding
        cols_to_round = stats_df.select_dtypes(include=[np.number]).columns
        stats_df[cols_to_round] = stats_df[cols_to_round].round(2)
        
        st.dataframe(stats_df.style.background_gradient(cmap='viridis'), use_container_width=True)
    else:
        st.info("No numeric columns available for statistics.")

    # Categorical Stats
    cat_df = df.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        st.markdown("#### Categorical Features")
        for col in cat_df.columns:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

def render_analysis(df, target, problem_type):
    """Render comprehensive data analysis tab."""
    st.markdown('<p class="subheader">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # 1. Target Distribution
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 🎯 Target Variable Distribution")
    
    if problem_type == "Classification":
        counts_df = df[target].value_counts().reset_index()
        counts_df.columns = [target, 'count']
        fig = px.pie(counts_df, names=target, values='count', hole=0.4, template="plotly_dark", title=f"Distribution of {target}")
    else:
        fig = px.histogram(df, x=target, nbins=30, marginal="box", template="plotly_dark", title=f"Distribution of {target}")
        
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 2. Correlation Analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Advanced Visualizations
    st.markdown("### 📈 Advanced Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot (2D/3D)", "Box Plot", "Violin Plot", "Pair Plot", "Histogram", "Line Plot"]
    )
    
    if viz_type == "Scatter Plot (2D/3D)":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_col = st.selectbox("X-Axis", df.columns)
        with col2:
            y_col = st.selectbox("Y-Axis", df.columns, index=1 if len(df.columns) > 1 else 0)
        with col3:
            z_col = st.selectbox("Z-Axis (Optional)", [None] + list(df.columns))
        with col4:
            color_col = st.selectbox("Color By", [None] + list(df.columns), index=list(df.columns).index(target) if target in df.columns else 0)
            
        if z_col:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, template="plotly_dark")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, template="plotly_dark", trendline="ols" if problem_type=="Regression" else None)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Feature (Y-Axis)", df.select_dtypes(include=[np.number]).columns)
        with col2:
            x_col = st.selectbox("Group By (X-Axis)", [None] + list(df.select_dtypes(include=['object', 'category']).columns))
            
        fig = px.box(df, y=y_col, x=x_col, color=x_col if x_col else None, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Violin Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Feature (Y-Axis)", df.select_dtypes(include=[np.number]).columns)
        with col2:
            x_col = st.selectbox("Group By (X-Axis)", [None] + list(df.select_dtypes(include=['object', 'category']).columns))
            
        fig = px.violin(df, y=y_col, x=x_col, color=x_col if x_col else None, box=True, points="all", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Pair Plot":
        st.info("Pair plots can be slow for large datasets. Selecting top 5 numeric features.")
        cols = st.multiselect("Select Features", df.select_dtypes(include=[np.number]).columns, default=list(df.select_dtypes(include=[np.number]).columns)[:4])
        if cols:
            fig = px.scatter_matrix(df, dimensions=cols, color=target if target in df.columns else None, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
    elif viz_type == "Histogram":
        col_hist = st.selectbox("Select Feature", df.columns)
        fig = px.histogram(df, x=col_hist, color=target if target in df.columns else None, nbins=30, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Line Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-Axis", df.columns)
        with col2:
            y_col = st.selectbox("Y-Axis", df.select_dtypes(include=[np.number]).columns)
        fig = px.line(df, x=x_col, y=y_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

def render_model_evaluation(y_test, model_predictions, model_probs, problem_type):
    """Render advanced model evaluation plots."""
    st.markdown("### 📊 Advanced Model Evaluation")
    
    if problem_type == "Classification":
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Confusion Matrix
        st.write("#### Confusion Matrix")
        selected_model_cm = st.selectbox("Select Model for Confusion Matrix", list(model_predictions.keys()))
        if selected_model_cm:
            cm = confusion_matrix(y_test, model_predictions[selected_model_cm])
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', 
                               title=f"Confusion Matrix - {selected_model_cm}", template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # ROC Curve (only for binary classification or multiclass with OneVsRest)
        # Simplified for binary or taking first class for now to avoid complexity in this snippet
        if model_probs and any(v is not None for v in model_probs.values()):
            st.write("#### ROC Curve")
            fig_roc = go.Figure()
            for model_name, probs in model_probs.items():
                if probs is not None:
                    # Handle binary classification
                    if probs.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {roc_auc:.2f})'))
            
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
            st.plotly_chart(fig_roc, use_container_width=True)
            
    else: # Regression
        st.write("#### Actual vs Predicted")
        selected_model_reg = st.selectbox("Select Model", list(model_predictions.keys()))
        if selected_model_reg:
            df_res = pd.DataFrame({'Actual': y_test, 'Predicted': model_predictions[selected_model_reg]})
            fig_reg = px.scatter(df_res, x='Actual', y='Predicted', title=f"Actual vs Predicted - {selected_model_reg}", 
                                 trendline="ols", template="plotly_dark")
            st.plotly_chart(fig_reg, use_container_width=True)
            
            # Residuals
            df_res['Residuals'] = df_res['Actual'] - df_res['Predicted']
            fig_resid = px.histogram(df_res, x='Residuals', title=f"Residuals Distribution - {selected_model_reg}", 
                                     nbins=30, template="plotly_dark")
            st.plotly_chart(fig_resid, use_container_width=True)
