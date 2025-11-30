# -------------------------------
# Tab 3: Model Training
# -------------------------------
with tabs[2]:
    st.markdown('<p class="subheader">Model Training & Evaluation</p>', unsafe_allow_html=True)

    # Model Selection
    st.write(f"**Problem Type (Debug):** {problem_type}")
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
        default_models = ["Logistic Regression", "Random Forest"] # Default models for classification
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
        default_models = ["Linear Regression", "Random Forest"] # Default models for regression

    # Let user select models - Manual Selection Only Now
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Select Models to Train")

    selected_models = []

    # Create a more visual model selection interface - ALWAYS MANUAL NOW
    model_cols = st.columns(2)
    for i, (model_name, model) in enumerate(model_dict.items()):
        with model_cols[i % 2]:
            # Default to first 2 models being checked - ALWAYS MANUAL NOW
            model_selected = st.checkbox(
                model_name,
                value=(i < 2), # Default to first 2 models being checked - ALWAYS MANUAL
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

    # ... rest of the Tab 3 code ... 