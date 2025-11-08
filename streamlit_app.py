import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching Flask templates design
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

# Flask API Configuration
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")

# Helper Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_current_data():
    """Fetch current AQI data from Flask API"""
    try:
        response = requests.get(f"{FLASK_API_URL}/api/current", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching current data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_forecast_data():
    """Fetch forecast data from Flask API"""
    try:
        response = requests.get(f"{FLASK_API_URL}/api/forecast", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching forecast: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_model_stats():
    """Fetch model statistics from Flask API"""
    try:
        response = requests.get(f"{FLASK_API_URL}/api/models/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching model stats: {str(e)}")
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

# Sidebar
with st.sidebar:
    # Brand section matching Flask
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
            🌫️ AQI Predict
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
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🧠 Model Analytics", "📅 Forecast"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Status badge
    st.markdown("""
    <div class="status-badge">
        <span class="status-dot"></span>
        <span>Live Data</span>
    </div>
    """, unsafe_allow_html=True)

# Main Content
if "📊 Dashboard" in page:
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Dashboard")
        st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Real-time air quality monitoring and predictions</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch current data
    current_data = fetch_current_data()
    
    if current_data:
        # Main AQI Card
        aqi = int(current_data.get('aqi', 0))
        category_data = current_data.get('category', {})
        category = category_data.get('level', 'Unknown')
        health_msg = category_data.get('health', 'No information available')
        color = category_data.get('color', '#667eea')
        
        display_aqi_card(aqi, category, color, health_msg)
        
        # Pollutants Grid
        st.markdown("### 💨 Current Pollutant Levels")
        st.markdown("<br>", unsafe_allow_html=True)
        
        pollutants = current_data.get('pollutants', {})
        
        # Create 2x3 grid for pollutants
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pm25 = pollutants.get('pm25', 'N/A')
            if pm25 != 'N/A':
                pm25 = f"{pm25:.1f}"
            display_pollutant_card("PM2.5", pm25, "µg/m³")
        
        with col2:
            pm10 = pollutants.get('pm10', 'N/A')
            if pm10 != 'N/A':
                pm10 = f"{pm10:.1f}"
            display_pollutant_card("PM10", pm10, "µg/m³")
        
        with col3:
            pm1 = pollutants.get('pm1', 'N/A')
            if pm1 != 'N/A':
                pm1 = f"{pm1:.1f}"
            display_pollutant_card("PM1", pm1, "µg/m³")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Additional Info
        location = current_data.get('location', {})
        city_name = location.get('city', 'N/A')
        dominant = location.get('dominant_pollutant', 'N/A').upper()
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
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">⚠️ Dominant Pollutant</div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{dominant}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("⚠️ Unable to fetch current data. Please check if Flask backend is running.")

elif "🧠 Model Analytics" in page:
    # Header
    st.title("🧠 Model Analytics")
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
        st.markdown("### 📈 Performance Metrics")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_r2_model = df['r2'].idxmax()
        best_mae_model = df['mae'].idxmin()
        
        with col1:
            st.metric("🏆 Best R² Score", 
                     f"{df['r2'].max():.4f}", 
                     delta=best_r2_model,
                     delta_color="off")
        
        with col2:
            st.metric("📉 Lowest MAE", 
                     f"{df['mae'].min():.4f}",
                     delta=best_mae_model,
                     delta_color="off")
        
        with col3:
            st.metric("📊 Models Trained", len(df))
        
        with col4:
            avg_r2 = df['r2'].mean()
            st.metric("📊 Average R²", f"{avg_r2:.4f}")
        
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
        st.markdown("### 📊 Model Comparison")
        
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
        st.markdown("### 📋 Detailed Performance Metrics")
        
        # Format the dataframe for display
        display_df = df[['r2', 'mae', 'rmse', 'rank', 'performance']].copy()
        display_df.columns = ['R² Score', 'MAE', 'RMSE', 'Rank', 'Performance']
        
        st.dataframe(
            display_df.style.background_gradient(cmap='RdYlGn', subset=['R² Score'])
                            .background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE'])
                            .format({'R² Score': '{:.4f}', 'MAE': '{:.4f}', 'RMSE': '{:.4f}'}),
            use_container_width=True,
            height=400
        )
    
    else:
        st.error("⚠️ Unable to fetch model statistics. Please ensure models have been trained.")

elif "📅 Forecast" in page:
    # Header
    st.title("📅 3-Day AQI Forecast")
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
        
        # Trend Chart
        st.markdown("### 📈 AQI Trend")
        
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
        st.error("⚠️ Unable to fetch forecast data. Please check if Flask backend is running.")

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
    🌍 AQI Prediction System | University of Karachi | Powered by ML & Real-time Data
</div>
""", unsafe_allow_html=True)
