import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error

# Try importing advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def get_models(problem_type):
    """Return a dictionary of available models based on problem type."""
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "k-Nearest Neighbors": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000),
            "Naive Bayes": GaussianNB()
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(eval_metric='logloss')
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier()
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostClassifier(verbose=0)
            
    else: # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
            "k-Nearest Neighbors": KNeighborsRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Neural Network": MLPRegressor(max_iter=1000)
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor()
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMRegressor()
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostRegressor(verbose=0)
            
    return models

def train_models(df, target, selected_models, problem_type, test_size, random_state, scaler_option):
    """Train selected models and return results, predictions, and probabilities."""
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('saved_models', exist_ok=True)
    
    model_dict = get_models(problem_type)
    results = []
    model_predictions = {}
    model_probs = {}
    trained_models_map = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(selected_models):
        if model_name not in model_dict:
            continue
            
        status_text.text(f"Training {model_name}...")
        model = model_dict[model_name]
        
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            model_predictions[model_name] = y_pred
            trained_models_map[model_name] = model
            
            # Get probabilities for classification
            if problem_type == "Classification":
                if hasattr(model, "predict_proba"):
                    model_probs[model_name] = model.predict_proba(X_test_scaled)
                else:
                    model_probs[model_name] = None
            
            # Save model
            model_path = os.path.join('saved_models', f'{model_name.replace(" ", "_")}.pkl')
            joblib.dump(model, model_path)
            
            if problem_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                results.append({
                    "Model": model_name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1
                })
            else:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results.append({
                    "Model": model_name,
                    "MSE": mse,
                    "MAE": mae,
                    "R2 Score": r2
                })
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(selected_models))
        
    status_text.text("Training complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return (pd.DataFrame(results).set_index("Model") if results else None, 
            model_predictions, 
            model_probs, 
            trained_models_map, 
            y_test,
            scaler)

def render_prediction_interface(df, target, trained_models, le_dict, scaler, problem_type):
    """Render the prediction interface."""
    st.markdown('<p class="subheader">Make Predictions</p>', unsafe_allow_html=True)
    
    if not trained_models:
        st.warning("Please train at least one model first.")
        return

    # Input Form
    st.markdown("### 📝 Input Features")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target]
        
        for i, col in enumerate(feature_cols):
            with col1 if i % 2 == 0 else col2:
                # Check if categorical (in le_dict)
                if col in le_dict:
                    # Get original classes
                    classes = le_dict[col].classes_
                    selected_val = st.selectbox(f"{col}", classes)
                    # Store encoded value for prediction
                    input_data[col] = le_dict[col].transform([selected_val])[0]
                else:
                    # Numeric input
                    val = st.number_input(f"{col}", value=float(df[col].mean()))
                    input_data[col] = val
                    
        submit_button = st.form_submit_button("🔮 Predict")
        
    if submit_button:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Scale input
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df
            
        st.markdown("### 🎯 Prediction Results")
        
        # Display predictions for all trained models
        results = []
        for model_name, model in trained_models.items():
            pred = model.predict(input_scaled)[0]
            
            if problem_type == "Classification":
                # Get probability if available
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_scaled)[0]
                    
                # Decode prediction if target was encoded
                # (Assuming target is encoded if it's in le_dict, but target is usually handled separately in load_dataset)
                # For now, we'll display the raw prediction or try to map it if we had the target encoder.
                # In data.py, target is not added to le_dict, but we can check if it's categorical.
                
                res = {"Model": model_name, "Prediction": pred}
                if prob is not None:
                    res["Probability"] = f"{max(prob):.2%}"
                results.append(res)
            else:
                results.append({"Model": model_name, "Prediction": f"{pred:.4f}"})
                
        st.dataframe(pd.DataFrame(results), use_container_width=True)
