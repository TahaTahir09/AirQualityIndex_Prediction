# AQI Predictor - Data Pipeline & ML Models

## Overview
Complete data collection, feature engineering, and machine learning pipeline for Air Quality Index (AQI) prediction. The system automatically adapts to available data and trains regression models optimized for small datasets.

## Quick Start

### 1. Check Your Data Status
```bash
python check_data_status.py
```

### 2. Collect Data
```bash
# Initial historical data collection
python backfill_pipeline.py

# Daily updates (schedule this)
python daily_pipeline.py
```

### 3. Train Models
```bash
python train_and_register_model.py
```
The system will automatically adapt to your data size and train the best model.

## Files

### Data Collection
- **data_fetcher.py** - Fetches data from AQICN API and processes features
- **feature_store.py** - Handles Hopsworks feature store operations
- **backfill_pipeline.py** - Collects 90 days of historical data
- **daily_pipeline.py** - Collects daily updates

### Model Training
- **train_and_register_model.py** - Adaptive ML training pipeline
- **check_data_status.py** - Data availability checker and recommender

### Configuration
- **config.py** - Configuration settings (API keys, horizons, etc.)

## Current Status

✓ **Model Trained**: RandomForest (R²=0.971, MAE=1.071)  
📊 **Data Available**: 27 rows (3.1 days)  
🎯 **Prediction Mode**: Single-step (next hour)  
📈 **Next Goal**: Collect 100+ rows for multi-day predictions

## Usage

### Initial Backfill (Run once)
```bash
python backfill_pipeline.py
```
Collects 90 days of historical data at 6-hour intervals (~360 samples).

### Daily Updates (Run daily or via cron/scheduler)
```bash
python daily_pipeline.py
```
Collects current data point and stores in Hopsworks.

### Model Training
```bash
python train_and_register_model.py
```

**Adaptive Features**:
- Automatically adjusts prediction horizon based on data availability
- Falls back to single-step prediction for small datasets (<50 rows)
- Optimized regression models for small data (Ridge, RandomForest)
- Smart train/test split based on dataset size

## Features Generated

### Raw Features
- AQI, PM2.5, PM10, O3, NO2, SO2, CO
- Temperature, Humidity, Pressure, Wind Speed, Dew Point

### Engineered Features
- Temporal: hour, day_of_week, day, month, year, is_weekend, time_of_day
- AQI category (0-6 scale)
- Interaction features: temp_humidity, wind_pm25, pressure_temp_ratio

## Data Storage
All data is stored in Hopsworks Feature Store:
- Feature Group: `aqi_features`
- Version: 1
- Primary Key: timestamp
- Event Time: datetime

## Configuration (config.py)

Key settings you can adjust:

```python
PREDICTION_HORIZON_DAYS = 1  # Days ahead to predict (auto-adjusts if needed)
MIN_TRAINING_ROWS = 20       # Minimum rows for training
BACKFILL_DAYS = 90           # Historical data to collect
COLLECTION_INTERVAL_HOURS = 1 # How often to collect data
```

## Data Requirements

| Prediction Goal | Min Rows | Recommended | Collection Time* |
|----------------|----------|-------------|-----------------|
| Next hour      | 20       | 50+         | 1-2 days        |
| 1 day ahead    | 50       | 100+        | 3-4 days        |
| 3 days ahead   | 100      | 200+        | 1 week          |
| 7 days ahead   | 200      | 500+        | 3 weeks         |

*With hourly data collection

## Models

The system trains and compares three regression models:

1. **RandomForest** (Usually best for AQI)
   - Handles non-linear patterns
   - Robust to outliers
   - Optimized: 50 trees, max_depth=5

2. **Ridge Regression** (Good for small data)
   - Fast and stable
   - Built-in regularization
   - Works well with limited samples

3. **Linear Regression** (Baseline)
   - Simple and interpretable
   - Good for linear trends

The best model is automatically selected and registered in Hopsworks.

## Troubleshooting

See `TRAINING_GUIDE.md` for detailed troubleshooting and optimization tips.
