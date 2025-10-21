import pandas as pd
import numpy as np
from feature_store import read_features
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Optional: Uncomment if you want to try deep learning
# import tensorflow as tf

WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", "088661c637816f9f1463ca3e44d37da6d739d021")
STATION_ID = os.getenv("STATION_ID", "A401143")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))


def fetch_data():
    df = read_features(
        api_key=HOPSWORKS_API_KEY,
        project_name=HOPSWORKS_PROJECT,
        feature_group_name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )
    return df


def prepare_data(df):
    # Drop rows with missing target
    df = df[df['aqi'].notna() & (df['aqi'] != -1)]
    # Features: drop columns not useful for prediction
    drop_cols = [
        'aqi', 'timestamp', 'timestamp_unix', 'station_name', 'station_url', 'datetime',
        'data_quality', 'total_imputed_features', 'year', 'month', 'day', 'hour',
        'aqi_category_numeric', 'aqi_category', 'discomfort_index'
    ]
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    # Drop all non-numeric columns (object, string, category)
    X = X.select_dtypes(include=[np.number])
    y = df['aqi']
    return X, y


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = evaluate(y_test, y_pred_rf)

    # Ridge Regression
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results['Ridge'] = evaluate(y_test, y_pred_ridge)

    # # TensorFlow Example (uncomment to use)
    # tf_model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     tf.keras.layers.Dense(1)
    # ])
    # tf_model.compile(optimizer='adam', loss='mse')
    # tf_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    # y_pred_tf = tf_model.predict(X_test).flatten()
    # results['TensorFlow'] = evaluate(y_test, y_pred_tf)

    # Select best model
    best_model_name = min(results, key=lambda k: results[k][0])  # lowest RMSE
    if best_model_name == 'RandomForest':
        best_model = rf
    elif best_model_name == 'Ridge':
        best_model = ridge
    # elif best_model_name == 'TensorFlow':
    #     best_model = tf_model
    else:
        best_model = rf

    print("\nModel Performance:")
    for name, (rmse, mae, r2) in results.items():
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
    print(f"\nBest model: {best_model_name}")

    return best_model, best_model_name, results


def register_model(model, model_name, metrics):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    model_obj = mr.python.create_model(
        name=model_name,
        metrics={
            "rmse": metrics[0],
            "mae": metrics[1],
            "r2": metrics[2]
        },
        description=f"{model_name} for AQI prediction",
        input_example=None
    )
    model_obj.save(model_dir)
    print(f"\n✓ Model '{model_name}' registered in Hopsworks Model Registry.")


def main():
    print("\n=== AQI Model Training and Registration ===\n")
    df = fetch_data()
    if df is None or len(df) < 10:
        print("✗ Not enough data for training.")
        return
    X, y = prepare_data(df)
    model, model_name, results = train_and_evaluate(X, y)
    register_model(model, model_name, results[model_name])


if __name__ == "__main__":
    main()
