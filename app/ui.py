import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Machine Learning Playground",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
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
    
    /* Headers */
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
    
    /* Metric Cards */
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
    
    /* Tabs */
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
    
    /* Inputs */
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
    
    /* Sidebar */
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #BB86FC;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #7B1FA2;
    }
    
    /* Buttons */
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
    
    /* Text fixes */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    .stButton > button span {
        color: white !important;
    }
    
    .streamlit-expanderHeader {
        color: #FFFFFF;
        font-weight: 500;
    }
    
    .stRadio > div, .stMultiSelect > div, .stSlider > div, 
    .stCheckbox > label, .stSelectbox > div > div {
        color: #FFFFFF;
    }
    
    .stNumberInput > div > div > input {
        color: #FFFFFF;
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    .stSelectbox > div > div {
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    .stMultiSelect > div {
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
    }
    
    /* DataFrame */
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

def render_header():
    st.markdown('<h1 class="main-header">🧪 Machine Learning Playground</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #BB86FC;">
            Explore, train, and evaluate machine learning models with an intuitive interface.
            Select options from the sidebar!
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_header():
    st.sidebar.image("assets/ml-playground.png", width=300)
    st.sidebar.markdown('<p class="sidebar-header">ML Playground Configuration</p>', unsafe_allow_html=True)

def render_footer():
    st.sidebar.markdown("---")
    st.sidebar.caption("ML PlayGround | by faizy")
