WAQI_API_TOKEN = "088661c637816f9f1463ca3e44d37da6d739d021"
STATION_ID = "A401143"
HOPSWORKS_API_KEY = "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5"
HOPSWORKS_PROJECT = "AQI_Project_10"

BACKFILL_DAYS = 90
COLLECTION_INTERVAL_HOURS = 1

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
 
# How many days ahead to predict (integer)
# Set to 1 for small datasets (< 100 rows), can increase with more data
PREDICTION_HORIZON_DAYS = 3

# Minimum rows required for training (will auto-adjust horizon if needed)
MIN_TRAINING_ROWS = 20
