# AQI Predictor - Data Pipeline

## Overview
Data collection and feature engineering pipeline for Air Quality Index (AQI) prediction.

## Files

- **data_fetcher.py** - Fetches data from AQICN API and processes features
- **feature_store.py** - Handles Hopsworks feature store operations
- **backfill_pipeline.py** - Collects 90 days of historical data
- **daily_pipeline.py** - Collects daily updates
- **config.py** - Configuration settings

## Usage

### Initial Backfill (Run once)
```bash
python backfill_pipeline.py
```
Collects 90 days of historical data at 6-hour intervals (~360 samples).

### Daily Updates (Run daily)
```bash
python daily_pipeline.py
```
Collects current data point and stores in Hopsworks.

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
