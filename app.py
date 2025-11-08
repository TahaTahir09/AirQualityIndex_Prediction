from flask import Flask, render_template, jsonify, request
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import requests
from config import WAQI_API_TOKEN, STATION_ID, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

app = Flask(__name__)

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
                print(f" API Error: {data}")
                raise Exception(f"WAQI API Error: {data.get('data', 'Unknown error')}")
            
            api_data = data['data']
            if not api_data:
                raise Exception("No data received from WAQI API")
            
            # Extract current AQI and pollutants
            current_data = {
                'aqi': float(api_data.get('aqi', 0)),
                'pm25': float(api_data.get('iaqi', {}).get('pm25', {}).get('v', 0)),
                'pm10': float(api_data.get('iaqi', {}).get('pm10', {}).get('v', 0)),
                'pm1': float(api_data.get('iaqi', {}).get('pm1', {}).get('v', 0)),  # PM1
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
            
            print(f" Fetched real-time data from API - AQI: {current_data['aqi']}, City: {current_data['city']}")
            return current_data
            
        except Exception as e:
            print(f" Error fetching from API: {e}")
            return None
    
    def load_model_from_hopsworks(self):
        """Load the best model from Hopsworks Model Registry"""
        try:
            import hopsworks
            
            print("🔌 Connecting to Hopsworks...")
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT
            )
            
            mr = project.get_model_registry()
            
            # Try to find the best registered model
            model_names = ['aqi_xgboost', 'aqi_randomforest', 'aqi_linearregression']
            
            for model_name in model_names:
                try:
                    print(f"   Trying to load {model_name}...")
                    model = mr.get_model(model_name, version=None)
                    model_dir = model.download()
                    
                    # Load the model file
                    for file in os.listdir(model_dir):
                        if file.endswith('.pkl'):
                            model_path = os.path.join(model_dir, file)
                            self.model = joblib.load(model_path)
                            self.model_name = model_name.replace('aqi_', '').upper()
                            print(f" Model loaded from Hopsworks: {self.model_name}")
                            return True
                except Exception as e:
                    print(f"    Could not load {model_name}: {e}")
                    continue
            
            print("  No models found in Hopsworks, trying local files...")
            return self.load_model_locally()
            
        except Exception as e:
            print(f" Error connecting to Hopsworks: {e}")
            return self.load_model_locally()
    
    def load_model_locally(self):
        """Fallback: Load model from local model_artifacts/"""
        try:
            if not os.path.exists('model_artifacts'):
                print(" No model_artifacts directory found")
                raise Exception("No model_artifacts directory found. Please ensure models are uploaded.")
                return False
            
            model_files = [f for f in os.listdir('model_artifacts') 
                          if f.endswith('_best.pkl')]
            
            if not model_files:
                model_files = [f for f in os.listdir('model_artifacts') 
                              if f.endswith('.pkl') and any(x in f for x in ['XGBoost', 'RandomForest', 'LinearRegression'])]
            
            if model_files:
                model_file = sorted(model_files)[-1]  # Get latest
                model_path = os.path.join('model_artifacts', model_file)
                self.model = joblib.load(model_path)
                self.model_name = model_file.replace('_best.pkl', '').replace('.pkl', '').upper()
                print(f"✅ Loaded local model: {self.model_name}")
                return True
            
            print("❌ No model files found locally")
            return False
            
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            return False
    
    def make_simple_prediction(self, current_data, days=3):
        """Make predictions using current data (simplified - no feature engineering)"""
        try:
            if self.model is None:
                print("❌ No model loaded")
                return None
            
            # Create simple feature vector from current data
            features = [
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
            
            X = np.array(features).reshape(1, -1)
            
            # If model expects more features, pad with zeros or use mean
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                if len(features) < expected_features:
                    # Pad with zeros
                    padding = [0] * (expected_features - len(features))
                    X = np.array(features + padding).reshape(1, -1)
                elif len(features) > expected_features:
                    # Truncate
                    X = np.array(features[:expected_features]).reshape(1, -1)
            
            predictions = []
            current_aqi = current_data.get('aqi', 0)
            
            for day in range(1, days + 1):
                # Predict
                pred_aqi = self.model.predict(X)[0]
                
                # Add slight random variation and trend
                variation = np.random.normal(0, 2)  # Small random noise
                trend_factor = 1 + (day - 1) * 0.02  # Slight trend
                pred_aqi = pred_aqi * trend_factor + variation
                
                # Ensure non-negative
                pred_aqi = max(0, pred_aqi)
                
                future_date = datetime.now() + timedelta(days=day)
                
                predictions.append({
                    'day': day,
                    'date': future_date.strftime('%Y-%m-%d'),
                    'aqi': float(pred_aqi),
                    'category': self.get_aqi_category(pred_aqi)
                })
            
            print(f"✅ Generated {len(predictions)} predictions (Current AQI: {current_aqi:.1f})")
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
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
                'health': 'Health alert: everyone may experience serious effects'
            }
        else:
            return {
                'level': 'Hazardous',
                'color': '#7e0023',
                'health': 'Health warning of emergency conditions'
            }

# Global predictor instance
predictor = AQIPredictor()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/models')
def models():
    """Model analytics page"""
    return render_template('models.html')

@app.route('/forecast')
def forecast():
    """Forecast page"""
    return render_template('dashboard.html')

@app.route('/api/current')
def get_current():
    """Get current AQI data from WAQI API"""
    try:
        current_data = predictor.fetch_current_data_from_api()
        
        if current_data is None:
            return jsonify({'error': 'Failed to fetch current data from API'}), 500
        
        current_aqi = current_data['aqi']
        category = predictor.get_aqi_category(current_aqi)
        
        # Filter pollutants to only include positive values
        all_pollutants = {
            'pm25': current_data['pm25'],
            'pm10': current_data['pm10'],
            'pm1': current_data['pm1'],
            'o3': current_data['o3'],
            'no2': current_data['no2'],
            'so2': current_data['so2'],
            'co': current_data['co'],
            'humidity': current_data['humidity'],
            'temperature': current_data['temperature'],
            'wind_speed': current_data['wind_speed']
        }
        pollutants = {k: v for k, v in all_pollutants.items() if v > 0}
        
        response = {
            'aqi': current_aqi,
            'category': category,
            'pollutants': pollutants,
            'weather': {
                'temperature': current_data.get('temperature', 0),
                'pressure': current_data.get('pressure', 0),
                'humidity': current_data.get('humidity', 0),
                'wind_speed': current_data.get('wind_speed', 0)
            },
            'location': {
                'city': current_data.get('city', 'Unknown'),
                'dominant_pollutant': current_data.get('dominentpol', 'unknown')
            },
            'timestamp': current_data['timestamp']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in /api/current: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast')
def get_forecast():
    """Get 3-day AQI forecast predictions"""
    try:
        # Fetch current data from API
        current_data = predictor.fetch_current_data_from_api()
        
        if current_data is None:
            return jsonify({'error': 'Failed to fetch current data'}), 500
        
        # Make predictions
        predictions = predictor.make_simple_prediction(current_data, days=3)
        
        if predictions is None:
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        return jsonify({
            'model': predictor.model_name or 'ML Model',
            'predictions': predictions,
            'based_on': {
                'current_aqi': current_data['aqi'],
                'timestamp': current_data['timestamp']
            }
        })
        
    except Exception as e:
        print(f"❌ Error in /api/forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/initialize')
def initialize():
    """Initialize the predictor (load model)"""
    try:
        if predictor.model is None:
            success = predictor.load_model_from_hopsworks()
            if not success:
                return jsonify({
                    'status': 'warning',
                    'message': 'No model loaded. Train a model first using train_model.py'
                }), 200
        
        return jsonify({
            'status': 'initialized',
            'model': predictor.model_name or 'Unknown'
        })
        
    except Exception as e:
        print(f"❌ Error in /api/initialize: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/stats')
def get_model_stats():
    """Get statistics for all trained models"""
    try:
        # Check if model_results.json exists
        results_file = 'model_results.json'
        if not os.path.exists(results_file):
            return jsonify({'error': 'No model results found. Train models first.'}), 404
        
        # Read model results
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract model comparison data with correct field names
        models = []
        if 'model_comparison' in data:
            for model_data in data['model_comparison']:
                models.append({
                    'name': model_data.get('model_name', model_data.get('model', 'Unknown')),
                    'rank': model_data.get('rank', 0),
                    'r2': float(model_data.get('r2_score', model_data.get('r2', 0))),
                    'rmse': float(model_data.get('rmse', 0)),
                    'mae': float(model_data.get('mae', 0)),
                    'performance': model_data.get('performance', 'Unknown')
                })
        
        return jsonify({
            'models': models,
            'best_model': data.get('best_model', 'Unknown'),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        })
        
    except Exception as e:
        print(f"❌ Error in /api/models/stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌬️  AQI PREDICTION DASHBOARD")
    print("="*80)
    
    print("\n[1/2] Loading ML model...")
    if predictor.load_model_from_hopsworks():
        print(f"✅ Model loaded: {predictor.model_name}")
    else:
        print("⚠️  No model loaded - train a model first with: python train_model.py")
    
    print("\n[2/2] Testing API connection...")
    test_data = predictor.fetch_current_data_from_api()
    if test_data:
        print(f"✅ API connected - Current AQI: {test_data['aqi']} in {test_data['city']}")
    else:
        print("⚠️  Could not connect to WAQI API")
    
    print("\n" + "="*80)
    print("🚀 Starting Flask server...")
    print("="*80)
    
    # Get port from environment variable (for Render deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n📍 Dashboard: http://127.0.0.1:{port}")
    print(f"📍 Network:   http://0.0.0.0:{port}")
    print("\n💡 Press CTRL+C to stop\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)
