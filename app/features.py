import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
import plotly.express as px

def render_feature_engineering(df, target, problem_type):
    """Render feature engineering tab."""
    st.markdown('<p class="subheader">Feature Engineering</p>', unsafe_allow_html=True)
    
    df_engineered = df.copy()
    
    # 1. Scaling
    st.markdown("### 1. Feature Scaling")
    st.info("Scale numerical features to a standard range.")
    
    scale_cols = st.multiselect(
        "Select columns to scale",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=df.select_dtypes(include=[np.number]).columns.tolist()[:2]
    )
    
    scaler_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    
    if st.button("Apply Scaling"):
        if scale_cols:
            if scaler_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
                
            df_engineered[scale_cols] = scaler.fit_transform(df_engineered[scale_cols])
            st.success(f"Applied {scaler_method} to {len(scale_cols)} columns.")
            st.dataframe(df_engineered[scale_cols].head(), use_container_width=True)
        else:
            st.warning("Please select columns to scale.")

    st.markdown("---")

    # 2. Dimensionality Reduction (PCA)
    st.markdown("### 2. Dimensionality Reduction (PCA)")
    st.info("Reduce the number of dimensions while retaining variance.")
    
    pca_cols = st.multiselect(
        "Select columns for PCA",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=df.select_dtypes(include=[np.number]).columns.tolist()
    )
    
    n_components = st.slider("Number of Components", min_value=2, max_value=min(len(pca_cols), 10), value=2)
    
    if st.button("Apply PCA"):
        if len(pca_cols) >= 2:
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(df[pca_cols])
            
            # Create new dataframe with PC columns
            pca_df = pd.DataFrame(
                data=components, 
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            # Add target for visualization
            if target in df.columns:
                pca_df[target] = df[target].values
            
            st.success(f"PCA reduced dimensions from {len(pca_cols)} to {n_components}. Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Visualize 2D or 3D
            if n_components >= 3:
                fig = px.scatter_3d(
                    pca_df, x='PC1', y='PC2', z='PC3',
                    color=target if target in df.columns else None,
                    title="3D PCA Visualization",
                    template="plotly_dark"
                )
            else:
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    color=target if target in df.columns else None,
                    title="2D PCA Visualization",
                    template="plotly_dark"
                )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show loadings
            loadings = pd.DataFrame(
                pca.components_.T, 
                columns=[f'PC{i+1}' for i in range(n_components)], 
                index=pca_cols
            )
            st.write("PCA Loadings:")
            st.dataframe(loadings.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            
        else:
            st.warning("Select at least 2 columns for PCA.")

    st.markdown("---")

    # 3. Feature Selection
    st.markdown("### 3. Feature Selection")
    st.info("Select the most important features based on statistical tests.")
    
    k_features = st.slider("Select Top K Features", min_value=1, max_value=len(df.columns)-1, value=min(5, len(df.columns)-1))
    
    if st.button("Run Feature Selection"):
        X = df.drop(target, axis=1).select_dtypes(include=[np.number])
        y = df[target]
        
        if problem_type == "Classification":
            # Use f_classif for classification (ANOVA)
            selector = SelectKBest(score_func=f_classif, k=min(k_features, len(X.columns)))
        else:
            # Use f_regression for regression
            selector = SelectKBest(score_func=f_regression, k=min(k_features, len(X.columns)))
            
        try:
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices]
            scores = selector.scores_[selected_indices]
            
            st.success(f"Selected Top {len(selected_features)} Features:")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Feature': selected_features,
                'Score': scores
            }).sort_values(by='Score', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(results_df, use_container_width=True)
            with col2:
                fig = px.bar(results_df, x='Score', y='Feature', orientation='h', 
                             title="Feature Importance Scores", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in feature selection: {e}. Ensure data is numeric and contains no NaNs.")

    return df_engineered
