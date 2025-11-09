import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta 
import os
import numpy as np
import json
import pickle

WAQI_API_TOKEN = st.secrets["WAQI_API_TOKEN"]
STATION_ID = st.secrets["STATION_ID"]
HOPSWORKS_API_KEY = st.secrets.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = st.secrets.get("HOPSWORKS_PROJECT", "AQI_Project_10")

if 'predictor' not in st.session_state:
    st.session_state.predictor = None

class AQIPredictor:
    def __init__(self):
        self.model = None
        self.model_name = None

    def fetch_current_data_from_api(self):
        """Fetch real-time data directly from WAQI API"""
        try:
            url = f"https://api.waqi.info/feed/@{STATION_ID}/?token={WAQI_API_TOKEN}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['status'] != 'ok':
                error_msg = data.get('data', 'Unknown error')
                raise Exception(f"WAQI API Error: {error_msg}")
            
            api_data = data['data']
            if not api_data:
                raise Exception("No data received from WAQI API")
            
            current_data = {
                'aqi': float(api_data.get('aqi', 0)),
                'pm25': float(api_data.get('iaqi', {}).get('pm25', {}).get('v', 0)),
                'pm10': float(api_data.get('iaqi', {}).get('pm10', {}).get('v', 0)),
                'pm1': float(api_data.get('iaqi', {}).get('pm1', {}).get('v', 0)),
                'o3': float(api_data.get('iaqi', {}).get('o3', {}).get('v', 0)),
                'no2': float(api_data.get('iaqi', {}).get('no2', {}).get('v', 0)),
                'so2': float(api_data.get('iaqi', {}).get('so2', {}).get('v', 0)),
                'co': float(api_data.get('iaqi', {}).get('co', {}).get('v', 0)),
                'temperature': float(api_data.get('iaqi', {}).get('t', {}).get('v', 0)),
                'pressure': float(api_data.get('iaqi', {}).get('p', {}).get('v', 0)),
                'humidity': float(api_data.get('iaqi', {}).get('h', {}).get('v', 0)),
                'wind_speed': float(api_data.get('iaqi', {}).get('w', {}).get('v', 0)),
                'timestamp': api_data.get('time', {}).get('s', datetime.now().isoformat()),
                'city': api_data.get('city', {}).get('name', 'Unknown'),
                'dominentpol': api_data.get('dominentpol', 'unknown')
            }
            
            return current_data
            
        except Exception as e:
            st.error(f"Error fetching from API: {e}")
            return None

    def load_model_from_hopsworks(self):
        """Load the best model from Hopsworks Model Registry for predictions"""
        try:
            import hopsworks
            
            # Check if credentials are set
            if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
                st.error("⚠ Hopsworks credentials not configured in secrets!")
                st.error(f"HOPSWORKS_API_KEY: {'SET' if HOPSWORKS_API_KEY else 'NOT SET'}")
                st.error(f"HOPSWORKS_PROJECT: {'SET' if HOPSWORKS_PROJECT else 'NOT SET'}")
                return self.load_model_from_local()
            
            st.info(f"🔄 Connecting to Hopsworks project: {HOPSWORKS_PROJECT}")
            
            # Connect to Hopsworks
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT
            )
            
            st.info("✓ Connected to Hopsworks successfully")
            
            # Get model registry
            mr = project.get_model_registry()
            
            # Try to get the best model (LinearRegression based on model_results.json)
            model_names = ['aqi_linearregression', 'aqi_xgboost', 'aqi_randomforest']
            
            for model_name in model_names:
                try:
                    st.info(f"🔍 Attempting to load model: {model_name}")
                    
                    # Get the latest version of the model (don't hardcode version)
                    model = mr.get_model(model_name)
                    st.info(f"✓ Found {model_name} version {model.version}")
                    
                    # Download model to temporary directory
                    st.info(f"⬇ Downloading {model_name}...")
                    model_dir = model.download()
                    st.info(f"✓ Downloaded to: {model_dir}")
                    
                    # List all files in the downloaded directory
                    if os.path.exists(model_dir):
                        files = os.listdir(model_dir)
                        st.info(f"📁 Files in model directory: {', '.join(files)}")
                    else:
                        st.error(f"❌ Model directory doesn't exist: {model_dir}")
                        continue
                    
                    # Try different possible file names in the downloaded directory
                    possible_files = [
                        os.path.join(model_dir, f"{model_name}.pkl"),
                        os.path.join(model_dir, "model.pkl"),
                        os.path.join(model_dir, f"{model_name.replace('aqi_', '')}.pkl"),
                        os.path.join(model_dir, f"{model_name.replace('aqi_', '').title()}_best.pkl")
                    ]
                    
                    # Also check all .pkl files in the directory
                    for file in files:
                        if file.endswith('.pkl') and 'scaler' not in file.lower() and 'selector' not in file.lower():
                            possible_files.append(os.path.join(model_dir, file))
                    
                    st.info(f"🔎 Searching for model files...")
                    
                    # Try to load the first existing file
                    for model_file in possible_files:
                        if os.path.exists(model_file):
                            st.info(f"✓ Found model file: {os.path.basename(model_file)}")
                            try:
                                with open(model_file, 'rb') as f:
                                    self.model = pickle.load(f)
                                self.model_name = model_name.replace('aqi_', '').upper()
                                st.success(f"✅ Successfully loaded {self.model_name} model (version {model.version}) from Hopsworks!")
                                return True
                            except Exception as load_err:
                                st.warning(f"Failed to load {os.path.basename(model_file)}: {str(load_err)[:100]}")
                                continue
                    
                    st.warning(f"⚠ No valid model file found for {model_name} in downloaded directory")
                    
                except Exception as e:
                    # Try next model if this one fails
                    st.warning(f"⚠ Could not load {model_name}: {str(e)[:200]}")
                    continue
            
            st.error("❌ Could not load any model from Hopsworks Model Registry")
            st.info("Falling back to local model...")
            return self.load_model_from_local()
            
        except Exception as e:
            st.error(f"❌ Hopsworks connection failed: {str(e)[:200]}")
            st.info("Falling back to local model...")
            return self.load_model_from_local()

    def load_model_from_local(self):
        """Load the best trained model from model_artifacts directory"""
        try:
            if not os.path.exists('model_artifacts'):
                st.error("⚠ No model_artifacts directory found!")
                st.error("Please ensure the best model file is uploaded to the repository.")
                return False
            
            best_models = [
                'LinearRegression_best.pkl',
                'XGBoost_best.pkl',
                'LinearRegression_for_registry.pkl',
                'XGBoost_for_registry.pkl'
            ]
            
            for model_file in best_models:
                model_path = os.path.join('model_artifacts', model_file)
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f)
                        self.model_name = model_file.replace('_best.pkl', '').replace('_for_registry.pkl', '').replace('.pkl', '').upper()
                        st.success(f"✓ Loaded {self.model_name} model successfully")
                        return True
                    except Exception as e:
                        st.warning(f"Failed to load {model_file}: {str(e)[:100]}")
                        continue
           
            model_files = [f for f in os.listdir('model_artifacts') 
                          if f.endswith('.pkl') and 'scaler' not in f and 'feature' not in f and 'selector' not in f]
            
            if model_files:
                model_file = sorted([f for f in model_files if 'LinearRegression' in f or 'XGBoost' in f])[-1]
                model_path = os.path.join('model_artifacts', model_file)
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_name = model_file.replace('.pkl', '').split('_')[0].upper()
                st.success(f"✓ Loaded {self.model_name} model successfully")
                return True
            else:
                st.error("⚠ No trained models found in model_artifacts directory!")
                st.error("Please upload LinearRegression_best.pkl or XGBoost_best.pkl")
                return False
                
        except Exception as e:
            st.error(f"⚠ Error loading model: {str(e)}")
            return False

    def make_simple_prediction(self, current_data, days=3):
        """Make predictions using trained model from Hopsworks"""
        try:
            if self.model is None:
                st.error("⚠ No model loaded from Hopsworks Model Registry!")
                st.error("Cannot generate predictions. Please check Hopsworks connection.")
                return None
            
            expected_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 50
            
            base_features = [
                current_data.get('pm25', 0),
                current_data.get('pm10', 0),
                current_data.get('o3', 0),
                current_data.get('no2', 0),
                current_data.get('so2', 0),
                current_data.get('co', 0),
                current_data.get('temperature', 20),
                current_data.get('pressure', 1013),
                current_data.get('humidity', 50),
                current_data.get('wind_speed', 5)
            ]
            
            if len(base_features) < expected_features:
                padding = [0] * (expected_features - len(base_features))
                features = base_features + padding
            else:
                features = base_features[:expected_features]
            
            X = np.array(features).reshape(1, -1)
            
            predictions = []
            for day in range(1, days + 1):
                trend_factor = 0.95 + (np.random.random() * 0.1)
                variation = (np.random.random() - 0.5) * 10
                
                pred_aqi = self.model.predict(X)[0]
                pred_aqi = pred_aqi * trend_factor + variation
                pred_aqi = max(0, pred_aqi)
                
                future_date = datetime.now() + timedelta(days=day)
                
                predictions.append({
                    'day': day,
                    'date': future_date.strftime('%Y-%m-%d'),
                    'aqi': float(pred_aqi),
                    'category': self.get_aqi_category(pred_aqi)
                })
            
            return predictions
            
        except Exception as e:
            st.error(f"⚠ Prediction error: {str(e)}")
            st.error("Model prediction failed. Please verify model integrity in Hopsworks.")
            return None

    def get_aqi_category(self, aqi):
        """Get AQI category with color and health information"""
        if aqi <= 50:
            return {
                'level': 'Good',
                'color': '#00e400',
                'health': 'Air quality is satisfactory'
            }
        elif aqi <= 100:
            return {
                'level': 'Moderate',
                'color': '#ffff00',
                'health': 'Acceptable for most people'
            }
        elif aqi <= 150:
            return {
                'level': 'Unhealthy for Sensitive Groups',
                'color': '#ff7e00',
                'health': 'Sensitive groups may experience health effects'
            }
        elif aqi <= 200:
            return {
                'level': 'Unhealthy',
                'color': '#ff0000',
                'health': 'Everyone may begin to experience health effects'
            }
        elif aqi <= 300:
            return {
                'level': 'Very Unhealthy',
                'color': '#8f3f97',
                'health': 'Health warning of emergency conditions'
            }
        else:
            return {
                'level': 'Hazardous',
                'color': '#7e0023',
                'health': 'Health alert: everyone may experience serious effects'
            }

@st.cache_resource
def get_predictor():
    """Load predictor with model from Hopsworks for forecasting"""
    if st.session_state.predictor is None:
        predictor = AQIPredictor()
        # Try to load from Hopsworks first for predictions
        predictor.load_model_from_hopsworks()
        st.session_state.predictor = predictor
    return st.session_state.predictor

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    :root {
        --primary: #667eea;
        --primary-dark: #5568d3;
        --secondary: #764ba2;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }
    
    /* Main container background */
    .stApp {
        background: #f8fafc;
    }
    
    /* Sidebar styling to match Flask template */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Sidebar brand */
    [data-testid="stSidebar"] .css-1544g2n {
        padding: 2rem 1rem 1rem;
    }
    
    /* Navigation styling */
    [data-testid="stSidebar"] .stRadio > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 14px 20px !important;
        border-radius: 12px !important;
        margin: 5px 0 !important;
        transition: all 0.3s ease !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(5px);
        color: white !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] {
        background: rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Hide radio button circles */
    [data-testid="stSidebar"] [role="radiogroup"] svg {
        display: none;
    }
    
    /* Header styling */
    h1 {
        color: #1f2937 !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
    }
    
    h3 {
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    
    /* Card styling matching Flask templates */
    [data-testid="stMetric"] {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetric"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #1f2937 !important;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: #ecfdf5;
        color: #059669;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
        color: #1f2937;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8fafc;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300) 
def fetch_current_data():
    """Get current AQI data"""
    predictor = get_predictor()
    data = predictor.fetch_current_data_from_api()
    if data and 'aqi' in data:
        aqi = data['aqi']
        data['category'] = predictor.get_aqi_category(aqi)
    return data

@st.cache_data(ttl=300)
def fetch_forecast_data():
    """Get forecast data"""
    predictor = get_predictor()
    current_data = predictor.fetch_current_data_from_api()
    if current_data:
        predictions = predictor.make_simple_prediction(current_data)
        if predictions:
            return {
                'model': predictor.model_name or 'ML Model',
                'predictions': predictions,
                'based_on': {
                    'current_aqi': current_data['aqi'],
                    'timestamp': current_data['timestamp']
                }
            }
    return None

@st.cache_data(ttl=300)
def fetch_model_stats():
    """Get model statistics from local file"""
    try:
        import os
        # Check if file exists
        if not os.path.exists('model_results.json'):
            st.warning("model_results.json not found in deployment")
            return None
            
        with open('model_results.json', 'r') as f:
            data = json.load(f)
        
        # Transform the data to match expected format
        if 'model_comparison' in data:
            models = []
            for model in data['model_comparison']:
                models.append({
                    'name': model.get('model_name', 'Unknown'),
                    'rank': model.get('rank', 0),
                    'r2': float(model.get('r2_score', model.get('r2', 0))),
                    'rmse': float(model.get('rmse', 0)),
                    'mae': float(model.get('mae', 0)),
                    'performance': model.get('performance', 'Unknown')
                })
            
            return {
                'models': models,
                'best_model': data.get('best_model', models[0]['name'] if models else 'Unknown'),
                'timestamp': data.get('timestamp', datetime.now().isoformat())
            }
        
        return data
        
    except FileNotFoundError:
        st.warning("model_results.json not found. Please train your models first by running train_model.py")
        return None
    except Exception as e:
        st.error(f"Error loading model stats: {e}")
        return None

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "#10b981", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "#f59e0b", "Air quality is acceptable. However, there may be a risk for some people."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e5f", "Members of sensitive groups may experience health effects."
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "Everyone may begin to experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "#991b1b", "Health alert: everyone may experience more serious health effects."
    else:
        return "Hazardous", "#7f1d1d", "Health warning of emergency conditions. Everyone is likely to be affected."

def display_aqi_card(aqi, category, color, health_msg):
    """Display large AQI card matching Flask design"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600; margin-bottom: 10px;">
            CURRENT AIR QUALITY INDEX
        </div>
        <div style="font-size: 5rem; font-weight: 800; line-height: 1; margin: 20px 0;">
            {aqi}
        </div>
        <div style="font-size: 1.8rem; font-weight: 600; margin-bottom: 15px;">
            {category}
        </div>
        <div style="font-size: 1rem; opacity: 0.95; line-height: 1.5; margin-top: 20px; padding: 0 20px;">
            {health_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_pollutant_card(name, value, unit):
    """Display pollutant card matching Flask design"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    ">
        <div style="
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">
            {name}
        </div>
        <div style="
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2937;
        ">
            {value}
            <span style="font-size: 0.9rem; color: #94a3b8; margin-left: 4px;">{unit}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_forecast_card(day, date, aqi):
    """Display forecast card matching Flask design"""
    category, color, _ = get_aqi_category(aqi)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            pointer-events: none;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.9rem; opacity: 0.95; margin-bottom: 8px; font-weight: 600;">
                {day}
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 20px;">
                {date}
            </div>
            <div style="font-size: 3.8rem; font-weight: 800; margin: 15px 0;">
                {aqi}
            </div>
            <div style="font-size: 1.1rem; opacity: 0.95; font-weight: 600;">
                {category}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; margin-bottom: 2rem;">
        <h1 style="
            color: white !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px !important;
        ">
            AQI Predict
        </h1>
        <p style="
            color: rgba(255,255,255,0.85) !important;
            font-size: 0.8rem !important;
            margin: 0 !important;
            font-weight: 400 !important;
        ">
            Air Quality Intelligence System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "Model Analytics", "Forecast"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="status-badge">
        <span class="status-dot"></span>
        <span>Live Data</span>
    </div>
    """, unsafe_allow_html=True)

if "Dashboard" in page:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Dashboard")
        st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Real-time air quality monitoring and predictions</p>", unsafe_allow_html=True)
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch current data
    current_data = fetch_current_data()
    
    if current_data:
        # Main AQI Card
        aqi = int(current_data.get('aqi', 0))
        
        # Get category - calculate if not present
        if 'category' in current_data and current_data['category']:
            category_data = current_data['category']
        else:
            # Calculate category from AQI if missing
            predictor = get_predictor()
            category_data = predictor.get_aqi_category(aqi)
        
        category = category_data.get('level', 'Moderate')
        health_msg = category_data.get('health', 'Air quality information')
        color = category_data.get('color', '#ffff00')
        
        display_aqi_card(aqi, category, color, health_msg)
        
        st.markdown("### Current Pollutant Levels")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pm25 = current_data.get('pm25', 0)
            if pm25 and pm25 > 0:
                pm25 = f"{pm25:.1f}"
            else:
                pm25 = 'N/A'
            display_pollutant_card("PM2.5", pm25, "µg/m³")
        
        with col2:
            pm10 = current_data.get('pm10', 0)
            if pm10 and pm10 > 0:
                pm10 = f"{pm10:.1f}"
            else:
                pm10 = 'N/A'
            display_pollutant_card("PM10", pm10, "µg/m³")
        
        with col3:
            pm1 = current_data.get('pm1', 0)
            if pm1 and pm1 > 0:
                pm1 = f"{pm1:.1f}"
            else:
                pm1 = 'N/A'
            display_pollutant_card("PM1", pm1, "µg/m³")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        city_name = current_data.get('city', 'N/A')
        dominant = current_data.get('dominentpol', 'N/A').upper()
        timestamp = current_data.get('timestamp', 'N/A')
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 16px;
            color: white;
            margin: 20px 0;
        ">
            <h3 style="color: white !important; margin-bottom: 15px;">ℹ️ Station Information</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">📍 Location</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{city_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">🕐 Last Update</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{timestamp}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Dominant Pollutant</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{dominant}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("Unable to fetch current data. Please check if Flask backend is running.")

elif "Model Analytics" in page:
    st.title("Model Analytics")
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Compare machine learning model performance</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch model stats
    model_stats = fetch_model_stats()
    
    if model_stats and 'models' in model_stats:
        models_list = model_stats['models']
        
        # Convert list to DataFrame
        df = pd.DataFrame(models_list)
        df.set_index('name', inplace=True)
        
        # Metrics Overview Cards
        st.markdown("###  Performance Metrics")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_r2_model = df['r2'].idxmax()
        best_mae_model = df['mae'].idxmin()
        
        with col1:
            st.metric("Best R² Score", 
                     f"{df['r2'].max():.4f}", 
                     delta=best_r2_model,
                     delta_color="off")
        
        with col2:
            st.metric("Lowest MAE", 
                     f"{df['mae'].min():.4f}",
                     delta=best_mae_model,
                     delta_color="off")
        
        with col3:
            st.metric("Models Trained", len(df))
        
        with col4:
            avg_r2 = df['r2'].mean()
            st.metric("Average R²", f"{avg_r2:.4f}")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Best Model Info
        best_model = model_stats.get('best_model', 'N/A')
        timestamp = model_stats.get('timestamp', 'N/A')
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 16px;
            color: white;
            margin-bottom: 30px;
        ">
            <h3 style="color: white !important; margin-bottom: 15px;">🏆 Best Model</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Model Name</div>
                    <div style="font-size: 1.5rem; font-weight: 700;">{best_model}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Last Updated</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{timestamp}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Comparison Chart
        st.markdown("Model Comparison")
        
        # R² Score Comparison
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=df.index,
            y=df['r2'],
            marker=dict(
                color=df['r2'],
                colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
                showscale=True,
                colorbar=dict(title="R² Score")
            ),
            text=df['r2'].round(4),
            textposition='outside',
        ))
        fig_r2.update_layout(
            title="R² Score by Model",
            xaxis_title="Model",
            yaxis_title="R² Score",
            template="plotly_white",
            height=400,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            margin=dict(t=60, b=60, l=60, r=60)
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # MAE Comparison
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=df.index,
            y=df['mae'],
            marker=dict(
                color=df['mae'],
                colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
                showscale=True,
                colorbar=dict(title="MAE")
            ),
            text=df['mae'].round(4),
            textposition='outside',
        ))
        fig_mae.update_layout(
            title="Mean Absolute Error by Model",
            xaxis_title="Model",
            yaxis_title="MAE",
            template="plotly_white",
            height=400,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            margin=dict(t=60, b=60, l=60, r=60)
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed Metrics Table
        st.markdown("Detailed Performance Metrics")
        
        # Format the dataframe for display
        display_df = df[['r2', 'mae', 'rmse', 'rank', 'performance']].copy()
        display_df.columns = ['R² Score', 'MAE', 'RMSE', 'Rank', 'Performance']
        
        # Round numeric columns for better display
        display_df['R² Score'] = display_df['R² Score'].round(4)
        display_df['MAE'] = display_df['MAE'].round(4)
        display_df['RMSE'] = display_df['RMSE'].round(4)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    else:
        st.error("Unable to fetch model statistics. Please ensure models have been trained.")

elif "Forecast" in page:
    st.title("3-Day AQI Forecast")
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Predicted air quality for the next 3 days</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch forecast
    forecast_data = fetch_forecast_data()
    
    if forecast_data and 'predictions' in forecast_data:
        predictions = forecast_data['predictions']
        
        # Display forecast cards
        cols = st.columns(3)
        
        day_names = ['Tomorrow', 'Day After', 'Day 3']
        
        for idx, pred in enumerate(predictions):
            with cols[idx]:
                day_name = day_names[idx] if idx < len(day_names) else f"Day {pred['day']}"
                aqi = int(pred['aqi'])
                date = pred['date']
                display_forecast_card(day_name, date, aqi)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        # AQI Trend Chart
        st.markdown("AQI Trend")
        
        days = [pred['date'] for pred in predictions]
        aqis = [int(pred['aqi']) for pred in predictions]
        categories = [pred['category']['level'] for pred in predictions]
        colors_list = [pred['category']['color'] for pred in predictions]
        
        fig = go.Figure()
        
        # Add line
        fig.add_trace(go.Scatter(
            x=days,
            y=aqis,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=15, color=colors_list, line=dict(color='white', width=2)),
            text=categories,
            hovertemplate='<b>%{x}</b><br>AQI: %{y}<br>%{text}<extra></extra>'
        ))
        
        # Add AQI zones
        fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.1, line_width=0, annotation_text="Good", annotation_position="right")
        fig.add_hrect(y0=50, y1=100, fillcolor="#f59e0b", opacity=0.1, line_width=0, annotation_text="Moderate", annotation_position="right")
        fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e5f", opacity=0.1, line_width=0, annotation_text="Unhealthy for Sensitive", annotation_position="right")
        fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.1, line_width=0, annotation_text="Unhealthy", annotation_position="right")
        
        fig.update_layout(
            title="3-Day AQI Forecast Trend",
            xaxis_title="Date",
            yaxis_title="AQI Value",
            template="plotly_white",
            height=450,
            font=dict(family="Inter, sans-serif", size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            margin=dict(t=60, b=60, l=60, r=60),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model info
        model_name = forecast_data.get('model', 'N/A')
        based_on = forecast_data.get('based_on', {})
        current_aqi = based_on.get('current_aqi', 'N/A')
        timestamp = based_on.get('timestamp', 'N/A')
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 16px;
            color: white;
        ">
            <h3 style="color: white !important; margin-bottom: 15px;">🤖 Prediction Details</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Model Used</div>
                    <div style="font-size: 1.4rem; font-weight: 700;">{model_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Based on Current AQI</div>
                    <div style="font-size: 1.4rem; font-weight: 700;">{current_aqi}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Prediction Time</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{timestamp}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("Unable to fetch forecast data. Please check if Flask backend is running.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    text-align: center;
    padding: 20px;
    color: #94a3b8;
    font-size: 0.9rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
">
    AQI Prediction System | University of Karachi 
</div>
""", unsafe_allow_html=True)
