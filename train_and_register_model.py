import pandas as pd
import numpy as np
from feature_store import read_features
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from config import COLLECTION_INTERVAL_HOURS, PREDICTION_HORIZON_DAYS

# Optional: Uncomment if you want to try deep learning
# import tensorflow as tf

# Import MIN_TRAINING_ROWS if available
try:
    from config import MIN_TRAINING_ROWS
except ImportError:
    MIN_TRAINING_ROWS = 20

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


def prepare_data(df, adaptive_horizon=True):
    """
    Prepare features and targets for multi-step AQI prediction.
    
    Args:
        df: Input dataframe with AQI and pollutant data
        adaptive_horizon: If True, automatically reduce horizon if data is insufficient
    
    Returns:
        X: Feature matrix
        y: Target matrix (multi-step predictions)
        actual_horizon: The horizon days actually used (may be less than config if adaptive)
    """
    # Diagnostic counts
    original_len = len(df)

    # Drop rows with missing target
    df = df[df['aqi'].notna() & (df['aqi'] != -1)].copy()
    filtered_len = len(df)

    # Columns we don't want to use as features
    drop_cols = [
        'aqi', 'timestamp', 'timestamp_unix', 'station_name', 'station_url', 'datetime',
        'data_quality', 'total_imputed_features', 'year', 'month', 'day', 'hour',
        'aqi_category_numeric', 'aqi_category', 'discomfort_index'
    ]

    # Base features (numeric only)
    X_base = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    X_base = X_base.select_dtypes(include=[np.number])

    # Build multi-step targets for AQI and a small set of pollutant columns if present
    steps_per_day = max(1, int(24 / COLLECTION_INTERVAL_HOURS))
    horizon = PREDICTION_HORIZON_DAYS
    
    # Adaptive horizon: reduce if we don't have enough data
    if adaptive_horizon:
        max_shift = horizon * steps_per_day
        available_rows = filtered_len - max_shift
        
        # If we don't have enough rows, reduce the horizon
        while available_rows < MIN_TRAINING_ROWS and horizon > 0:
            horizon -= 1
            max_shift = horizon * steps_per_day
            available_rows = filtered_len - max_shift
            print(f"ℹ️  Reducing horizon to {horizon} day(s) due to limited data ({available_rows} rows available)")
        
        # If still not enough, try single-step prediction (next hour only)
        if available_rows < MIN_TRAINING_ROWS and horizon == 0:
            print(f"ℹ️  Using single-step (hourly) prediction due to very limited data")
            horizon = 0  # Will be handled as single-step below
    
    actual_horizon = horizon

    # choose additional features to predict if available
    pollutant_candidates = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    pollutant_cols = [c for c in pollutant_candidates if c in df.columns]

    target_cols = ['aqi'] + pollutant_cols

    y_frames = []
    y_col_names = []
    
    # Special case: single-step prediction (next reading only)
    if horizon == 0:
        shift_steps = 1  # Just predict the next reading
        for col in target_cols:
            col_name = f"{col}_t+1step"
            y_col_names.append(col_name)
            y_frames.append(df[col].shift(-shift_steps))
    else:
        # Multi-day prediction
        for h in range(1, horizon + 1):
            shift_steps = h * steps_per_day
            for col in target_cols:
                col_name = f"{col}_t+{h}d"
                y_col_names.append(col_name)
                y_frames.append(df[col].shift(-shift_steps))

    if not y_frames:
        raise ValueError("No target columns available for multi-step forecasting.")

    y = pd.concat(y_frames, axis=1)
    y.columns = y_col_names

    # Align X and y: drop rows with NaNs in y (because of shifting)
    valid_idx = y.dropna().index

    # If no valid rows remain after shifting, raise a diagnostic error with counts
    valid_rows = len(valid_idx)
    if valid_rows == 0:
        if horizon == 0:
            max_shift = 1
        else:
            max_shift = horizon * steps_per_day
        possible_rows_after_shift = max(0, filtered_len - max_shift)
        raise ValueError(
            f"Not enough rows for multi-step targets: original={original_len}, "
            f"after_filter={filtered_len}, steps_per_day={steps_per_day}, horizon_days={horizon}. "
            f"Shifting by up to {max_shift} rows leaves {possible_rows_after_shift} rows. "
            "Ensure you have more historical records (increase data range) or reduce the horizon/collection interval."
        )

    X = X_base.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    print(f"✓ Prepared {len(X)} samples with {X.shape[1]} features and {y.shape[1]} target outputs")
    if horizon == 0:
        print(f"  Prediction mode: Single-step (next reading)")
    else:
        print(f"  Prediction mode: Multi-day forecast ({horizon} day(s) ahead)")

    return X, y, actual_horizon


def evaluate(y_true, y_pred):
    # y_true and y_pred are 2D (n_samples, n_outputs)
    # Compute RMSE/MAE/R2 per output and return the mean across outputs
    # Protect shapes
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_outputs = y_true.shape[1]
    rmses = []
    maes = []
    r2s = []
    for i in range(n_outputs):
        rmses.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        maes.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        # r2_score can be ill-defined if constant; wrap in try/except
        try:
            r2s.append(r2_score(y_true[:, i], y_pred[:, i]))
        except Exception:
            r2s.append(float('nan'))

    # mean of metrics across outputs
    rmse_mean = float(np.nanmean(rmses))
    mae_mean = float(np.nanmean(maes))
    r2_mean = float(np.nanmean(r2s))
    return rmse_mean, mae_mean, r2_mean


def train_and_evaluate(X, y, horizon_days):
    """
    Train and evaluate multiple regression models.
    
    Args:
        X: Feature matrix
        y: Target matrix
        horizon_days: Number of days in prediction horizon (0 for single-step)
    
    Returns:
        best_model: The trained model with best performance
        best_model_name: Name of the best model
        results: Dictionary of all model performances
    """
    # Handle very small datasets
    if len(X) < 20:
        test_size = 0.15  # Use smaller test set
        print(f"ℹ️  Using smaller test set (15%) due to limited data")
    else:
        test_size = 0.2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    results = {}

    print(f"\nTraining models on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # Random Forest (supports multioutput, works well with small datasets)
    rf = RandomForestRegressor(
        n_estimators=50,  # Reduced for small datasets
        max_depth=5,  # Prevent overfitting on small datasets
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = evaluate(y_test, y_pred_rf)

    # Ridge Regression (multioutput, good for small datasets with regularization)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results['Ridge'] = evaluate(y_test, y_pred_ridge)

    # Linear Regression (simple baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['LinearRegression'] = evaluate(y_test, y_pred_lr)

    # # TensorFlow Example (uncomment to use)
    # tf_model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     tf.keras.layers.Dense(1)
    # ])
    # tf_model.compile(optimizer='adam', loss='mse')
    # tf_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    # y_pred_tf = tf_model.predict(X_test).flatten()
    # results['TensorFlow'] = evaluate(y_test, y_pred_tf)

    # Select best model by average RMSE across outputs
    best_model_name = min(results, key=lambda k: results[k][0])  # lowest mean RMSE
    if best_model_name == 'RandomForest':
        best_model = rf
    elif best_model_name == 'Ridge':
        best_model = ridge
    elif best_model_name == 'LinearRegression':
        best_model = lr
    else:
        best_model = rf

    print("\nModel Performance (mean across all outputs):")
    for name, (rmse, mae, r2) in results.items():
        marker = "★" if name == best_model_name else " "
        print(f"{marker} {name}: mean RMSE={rmse:.3f}, mean MAE={mae:.3f}, mean R2={r2:.3f}")
    print(f"\n✓ Best model: {best_model_name}")

    return best_model, best_model_name, results


def register_model(model, model_name, metrics, horizon_days):
    """
    Register the trained model in Hopsworks Model Registry.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        metrics: Tuple of (rmse, mae, r2) metrics
        horizon_days: Number of days in prediction horizon
    """
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    if horizon_days == 0:
        description = f"{model_name} for single-step (hourly) AQI and pollutant prediction"
    else:
        description = f"{model_name} for multi-step AQI and pollutant prediction ({horizon_days} day(s) ahead)"
    
    model_obj = mr.python.create_model(
        name=model_name,
        metrics={
            "mean_rmse": metrics[0],
            "mean_mae": metrics[1],
            "mean_r2": metrics[2],
            "horizon_days": horizon_days
        },
        description=description,
        input_example=None
    )
    model_obj.save(model_dir)
    print(f"\n✓ Model '{model_name}' registered in Hopsworks Model Registry.")
    print(f"  Horizon: {horizon_days} day(s)" if horizon_days > 0 else "  Mode: Single-step prediction")


def main():
    print("\n=== AQI Model Training and Registration ===\n")
    df = fetch_data()
    if df is None or len(df) < 10:
        print(f"✗ Not enough data for training (need at least 10 rows, got {len(df) if df is not None else 0}).")
        print("  Recommendation: Run backfill_pipeline.py or daily_pipeline.py to collect more data.")
        return
    
    print(f"✓ Loaded {len(df)} rows from Hopsworks Feature Store")
    
    try:
        X, y, actual_horizon = prepare_data(df, adaptive_horizon=True)
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        print("\n  Troubleshooting tips:")
        print("  1. Increase BACKFILL_DAYS in config.py to collect more historical data")
        print("  2. Reduce PREDICTION_HORIZON_DAYS in config.py (currently set to {})".format(PREDICTION_HORIZON_DAYS))
        print("  3. Run backfill_pipeline.py to populate more data")
        return

    # Ensure we have enough samples after shifting for multi-step forecasting
    min_samples = 10
    if X is None or len(X) < min_samples:
        print(f"✗ Not enough valid rows after preparing targets (need >= {min_samples}, got {0 if X is None else len(X)}).")
        print("  This can happen when your dataset is too small for the prediction horizon.")
        print("  The system attempted to auto-adjust, but still couldn't get enough data.")
        print("\n  Solutions:")
        print("  1. Collect more historical data (increase BACKFILL_DAYS)")
        print("  2. Wait for more data to accumulate over time")
        return

    model, model_name, results = train_and_evaluate(X, y, actual_horizon)
    register_model(model, model_name, results[model_name], actual_horizon)
    
    print("\n" + "="*60)
    print("✓ Training complete! Model is ready for predictions.")
    print("="*60)


if __name__ == "__main__":
    main()
