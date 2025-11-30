# machine-learning-playground

[![Author](https://img.shields.io/badge/Author-mohd--faizy-red?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)](https://github.com/mohd-faizy/Machine-Learning-Playground)

A full-stack, interactive machine learning playground built with **Streamlit**. Bridge the gap between theory and practice by exploring datasets, engineering features, training advanced models (XGBoost, LightGBM, CatBoost), and evaluating performance—all without writing code.

## demo

### ui preview

[![Try Live](https://img.shields.io/badge/Try%20Live-Click%20Here-28a745?style=for-the-badge)](https://xswbj4jfkurhmdxtriz9gw.streamlit.app)

![ML Playground UI](assets/ml-plyg.png)

### application workflow

1.  **Select Dataset & Problem Type** (Classification or Regression)
2.  **Choose ML Models** from scikit-learn
3.  **Train & Evaluate Models** using performance metrics
4.  **Compare Results & Make Predictions**

## quick start

```bash
git clone https://github.com/mohd-faizy/Machine-Learning-Playground.git
cd Machine-Learning-Playground

# using uv (recommended)
pip install uv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# run
streamlit run main.py
```

The application will launch at `http://localhost:8501`.

## features

- **Data**: Built-in datasets (Iris, Titanic, Penguins, Housing, etc.) + Custom CSV upload.
- **Preprocessing**: Robust scaling, missing value imputation, categorical encoding.
- **Feature Engineering**: PCA (2D/3D visualization), SelectKBest, Feature Importance.
- **Models**: Scikit-learn suite + Gradient Boosting (XGBoost, LightGBM, CatBoost).
- **Analysis**: Interactive Plotly/Seaborn charts, Correlation Heatmaps, Target Distribution.
- **Evaluation**: Confusion Matrix, ROC/AUC, Residuals, Learning Curves.

## models & datasets

| **Problem Type** | **Algorithms** | **Datasets** |
| :--- | :--- | :--- |
| **Classification** | Logistic Regression, Random Forest, SVM, k-NN, Naive Bayes, XGBoost, LightGBM, CatBoost | Iris, Titanic, Penguins, Breast Cancer, Wine, Digits |
| **Regression** | Linear Regression, Decision Tree, Random Forest, SVR, Gradient Boosting, MLP | Boston Housing, Diabetes, California Housing, Tips, Diamonds, Planets |

> **Note**: Custom CSV upload is supported for both problem types.

## file structure

```
Machine-Learning-Playground/
├── app/
│   ├── analysis.py    # visualization & metrics
│   ├── data.py        # loading & preprocessing
│   ├── features.py    # pca & selection
│   ├── models.py      # training logic
│   ├── ui.py          # streamlit config
│   └── utils.py       # helpers
├── assets/            # static resources
├── main.py            # entry point
└── requirements.txt   # dependencies
```

## license

MIT

## connect with me

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/F4izy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohd-faizy/)
[![Stack Exchange](https://img.shields.io/badge/Stack_Exchange-1E5397?style=for-the-badge&logo=stack-exchange&logoColor=white)](https://ai.stackexchange.com/users/36737/faizy)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)

</div>
