import pandas as pd
import numpy as np
from feature_store import read_features
import hopsworks
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime
from config import COLLECTION_INTERVAL_HOURS, PREDICTION_HORIZON_DAYS

# Advanced ML
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Hyperparameter Tuning
import optuna
from optuna.integration import TFKerasPruningCallback

# Import MIN_TRAINING_ROWS if available
try:
    from config import MIN_TRAINING_ROWS
except ImportError:
    MIN_TRAINING_ROWS = 20

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    Train and evaluate 5 advanced regression models with hyperparameter tuning.
    
    Models:
    1. Linear Regression (Baseline)
    2. Random Forest (Optimized)
    3. XGBoost (Best for tabular data)
    4. LSTM (Deep Learning for time series) 
    5. 1D CNN (Advanced deep learning)
    
    Args:
        X: Feature matrix
        y: Target matrix
        horizon_days: Number of days in prediction horizon (0 for single-step)
    
    Returns:
        best_model: The trained model with best performance
        best_model_name: Name of the best model
        results: Dictionary of all model performances
    """
    print("\n" + "="*80)
    print("TRAINING ADVANCED MODELS")
    print("="*80)
    
    # Handle very small datasets
    if len(X) < 20:
        test_size = 0.15
        print("ℹ️  Using smaller test set (15%) due to limited data")
    else:
        test_size = 0.2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    
    # Scale for deep learning models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # For sequential models, reshape to 3D
    sequence_length = min(24, len(X_train) // 4)  # Adaptive sequence length
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)
    
    results = {}
    models = {}
    
    optimize_hp = len(X_train) > 500  # Enable hyperparameter tuning for large datasets
    
    if optimize_hp:
        print(f"✓ Large dataset ({len(X_train)} samples) - Hyperparameter tuning ENABLED")
    else:
        print(f"  Dataset size: {len(X_train)} samples - Using default hyperparameters")
    
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Sequential data: {len(X_train_seq)} train, {len(X_test_seq)} test\n")
    
    # ========================================================================
    # MODEL 1: LINEAR REGRESSION (Baseline)
    # ========================================================================
    print("-" * 80)
    print("MODEL 1: Linear Regression (Baseline)")
    print("-" * 80)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['LinearRegression'] = evaluate(y_test, y_pred_lr)
    models['LinearRegression'] = lr
    
    print(f"  RMSE: {results['LinearRegression'][0]:.2f}")
    print(f"  MAE:  {results['LinearRegression'][1]:.2f}")
    print(f"  R²:   {results['LinearRegression'][2]:.4f}")
    
    # ========================================================================
    # MODEL 2: RANDOM FOREST (Optimized)
    # ========================================================================
    print("\n" + "-" * 80)
    print("MODEL 2: Random Forest Regressor")
    print("-" * 80)
    
    if optimize_hp:
        print("  Optimizing hyperparameters (Optuna)...")
        best_params = optimize_random_forest(X_train, y_train, n_trials=15)
        rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = evaluate(y_test, y_pred_rf)
    models['RandomForest'] = rf
    
    print(f"  RMSE: {results['RandomForest'][0]:.2f}")
    print(f"  MAE:  {results['RandomForest'][1]:.2f}")
    print(f"  R²:   {results['RandomForest'][2]:.4f}")
    
    # ========================================================================
    # MODEL 3: XGBOOST (Best for Tabular)
    # ========================================================================
    print("\n" + "-" * 80)
    print("MODEL 3: XGBoost Regressor")
    print("-" * 80)
    
    if optimize_hp:
        print("  Optimizing hyperparameters (Optuna)...")
        best_params = optimize_xgboost(X_train, y_train, n_trials=20)
        xgb_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    else:
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate(y_test, y_pred_xgb)
    models['XGBoost'] = xgb_model
    
    print(f"  RMSE: {results['XGBoost'][0]:.2f}")
    print(f"  MAE:  {results['XGBoost'][1]:.2f}")
    print(f"  R²:   {results['XGBoost'][2]:.4f}")
    
    # ========================================================================
    # MODEL 4: LSTM (Deep Learning)
    # ========================================================================
    if len(X_train_seq) > 50:  # Need enough sequential data
        print("\n" + "-" * 80)
        print("MODEL 4: LSTM (Deep Learning)")
        print("-" * 80)
        
        lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), y_train.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
        
        print(f"  Training with {lstm_model.count_params():,} parameters...")
        history = lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=100,
            batch_size=min(32, len(X_train_seq) // 4),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0)
        results['LSTM'] = evaluate(y_test_seq, y_pred_lstm)
        models['LSTM'] = lstm_model
        
        print(f"  RMSE: {results['LSTM'][0]:.2f}")
        print(f"  MAE:  {results['LSTM'][1]:.2f}")
        print(f"  R²:   {results['LSTM'][2]:.4f}")
        print(f"  Epochs: {len(history.history['loss'])}")
    else:
        print("\n⚠️  Skipping LSTM - insufficient sequential data (need >50 samples)")
    
    # ========================================================================
    # MODEL 5: 1D CNN (Advanced Deep Learning)
    # ========================================================================
    if len(X_train_seq) > 50:
        print("\n" + "-" * 80)
        print("MODEL 5: 1D CNN (Advanced Deep Learning)")
        print("-" * 80)
        
        cnn_model = build_cnn_model((X_train_seq.shape[1], X_train_seq.shape[2]), y_train.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
        
        print(f"  Training with {cnn_model.count_params():,} parameters...")
        history = cnn_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=100,
            batch_size=min(32, len(X_train_seq) // 4),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        y_pred_cnn = cnn_model.predict(X_test_seq, verbose=0)
        results['CNN_1D'] = evaluate(y_test_seq, y_pred_cnn)
        models['CNN_1D'] = cnn_model
        
        print(f"  RMSE: {results['CNN_1D'][0]:.2f}")
        print(f"  MAE:  {results['CNN_1D'][1]:.2f}")
        print(f"  R²:   {results['CNN_1D'][2]:.4f}")
        print(f"  Epochs: {len(history.history['loss'])}")
    else:
        print("\n⚠️  Skipping 1D CNN - insufficient sequential data (need >50 samples)")
    
    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL COMPARISON (sorted by RMSE)")
    print("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    for name, (rmse, mae, r2) in sorted_results:
        print(f"  {name:20s} | RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | R²: {r2:7.4f}")
    
    # Select best model
    best_model_name = min(results, key=lambda k: results[k][0])
    best_model = models[best_model_name]
    
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   RMSE: {results[best_model_name][0]:.2f}")
    print(f"   MAE:  {results[best_model_name][1]:.2f}")
    print(f"   R²:   {results[best_model_name][2]:.4f}")
    print("="*80)
    
    # Save all models
    save_all_models(models, scaler)
    
    return best_model, best_model_name, results


# Helper functions for advanced models

def create_sequences(X, y, sequence_length):
    """Create sequences for LSTM/CNN from tabular data"""
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def optimize_random_forest(X, y, n_trials=15):
    """Optuna optimization for Random Forest"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        y_pred = model.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgboost(X, y, n_trials=20):
    """Optuna optimization for XGBoost"""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 3),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X, y, verbose=False)
        y_pred = model.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_lstm_model(input_shape, output_dim):
    """Build LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_cnn_model(input_shape, output_dim):
    """Build 1D CNN model"""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def save_all_models(models, scaler):
    """Save all trained models"""
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, model in models.items():
        if 'LSTM' in name or 'CNN' in name:
            path = os.path.join(model_dir, f"{name}_{timestamp}.keras")
            model.save(path)
        else:
            path = os.path.join(model_dir, f"{name}_{timestamp}.pkl")
            joblib.dump(model, path)
        print(f"  ✓ Saved {name} to {path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler to {scaler_path}")


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
