"""
Exploratory Data Analysis (EDA) for AQI Prediction Project
===========================================================

This script performs comprehensive EDA on your 17,568+ rows of AQI data:
1. Data Overview & Quality Assessment
2. Statistical Summary & Distribution Analysis
3. Temporal Patterns & Trends
4. Correlation Analysis
5. Outlier Detection
6. Feature Relationships
7. Seasonality & Time Series Decomposition

Run this to understand your data before training models!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from feature_store import read_features
from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# ============================================================================
# 1. DATA OVERVIEW & QUALITY ASSESSMENT
# ============================================================================

def data_overview(df):
    """
    Provides initial understanding of dataset structure
    
    Techniques Applied:
    - Shape analysis: rows × columns
    - Data types inspection: identifying numeric vs categorical
    - Memory usage: understanding computational requirements
    - Column inventory: categorizing features by type
    """
    print("="*80)
    print("1. DATA OVERVIEW")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-"*80)
    print("Column Types:")
    print("-"*80)
    print(df.dtypes.value_counts())
    
    print("\n" + "-"*80)
    print("Feature Categories:")
    print("-"*80)
    
    # Categorize features
    pollutants = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    weather = ['temperature', 'humidity', 'pressure', 'wind_speed', 'dew_point']
    temporal = ['hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend', 'time_of_day_numeric']
    engineered = ['temp_humidity_interaction', 'discomfort_index', 'wind_pm25_interaction', 'pressure_temp_ratio']
    
    print(f"  • Pollutants: {len([c for c in pollutants if c in df.columns])} features")
    print(f"  • Weather: {len([c for c in weather if c in df.columns])} features")
    print(f"  • Temporal: {len([c for c in temporal if c in df.columns])} features")
    print(f"  • Engineered: {len([c for c in engineered if c in df.columns])} features")
    
    print(f"\nTotal Numeric Features: {df.select_dtypes(include=[np.number]).shape[1]}")


def data_quality_assessment(df):
    """
    Evaluates data completeness and quality issues
    
    Techniques Applied:
    - Missing value analysis: identifying gaps in data
    - Invalid value detection: finding -1 placeholders
    - Duplicate detection: checking for repeated records
    - Data completeness score: percentage of usable data
    """
    print("\n" + "="*80)
    print("2. DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    if missing.sum() > 0:
        print("\nMissing Values:")
        print("-"*80)
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Missing', ascending=False)
        print(missing_df)
    else:
        print("\n✓ No missing (NaN) values found!")
    
    # Invalid values (-1 placeholders)
    print("\n" + "-"*80)
    print("Invalid Values (marked as -1):")
    print("-"*80)
    
    pollutants = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    invalid_counts = {}
    
    for col in pollutants:
        if col in df.columns:
            invalid_count = (df[col] == -1).sum()
            if invalid_count > 0:
                invalid_counts[col] = invalid_count
    
    if invalid_counts:
        for col, count in invalid_counts.items():
            pct = (count / len(df) * 100)
            print(f"  • {col.upper()}: {count} rows ({pct:.2f}%)")
    else:
        print("  ✓ No invalid (-1) values in pollutant columns!")
    
    # Duplicates
    print("\n" + "-"*80)
    print("Duplicate Records:")
    print("-"*80)
    
    if 'timestamp' in df.columns:
        dup_count = df.duplicated(subset=['timestamp']).sum()
        print(f"  • Duplicate timestamps: {dup_count}")
    
    total_dups = df.duplicated().sum()
    print(f"  • Total duplicate rows: {total_dups}")
    
    # Data quality score
    valid_aqi = df[(df['aqi'].notna()) & (df['aqi'] != -1)]
    quality_score = (len(valid_aqi) / len(df) * 100)
    
    print("\n" + "-"*80)
    print(f"Data Quality Score: {quality_score:.2f}% usable records")
    print("-"*80)


# ============================================================================
# 2. STATISTICAL SUMMARY & DISTRIBUTION ANALYSIS
# ============================================================================

def statistical_summary(df):
    """
    Provides descriptive statistics for numerical features
    
    Techniques Applied:
    - Central tendency: mean, median (50th percentile)
    - Dispersion: standard deviation, min/max, quartiles
    - Distribution shape: identifying skewness via percentiles
    - Outlier indicators: values beyond typical ranges
    """
    print("\n" + "="*80)
    print("3. STATISTICAL SUMMARY")
    print("="*80)
    
    # Key pollutants and weather
    key_cols = ['aqi', 'pm25', 'pm10', 'o3', 'temperature', 'humidity', 'wind_speed', 'pressure']
    available_cols = [c for c in key_cols if c in df.columns]
    
    if available_cols:
        print("\nKey Metrics (Valid Data Only):")
        print("-"*80)
        
        # Filter valid data (exclude -1)
        valid_df = df[available_cols].replace(-1, np.nan)
        
        summary = valid_df.describe().T
        summary['skewness'] = valid_df.skew()
        summary['kurtosis'] = valid_df.kurtosis()
        
        # Format for readability
        pd.options.display.float_format = '{:.2f}'.format
        print(summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']])
        pd.options.display.float_format = None
        
        print("\n" + "-"*80)
        print("Distribution Interpretation:")
        print("-"*80)
        print("  • Skewness > 1: Highly right-skewed (long tail on right)")
        print("  • Skewness < -1: Highly left-skewed (long tail on left)")
        print("  • -0.5 < Skewness < 0.5: Approximately symmetric")


def distribution_analysis(df):
    """
    Analyzes AQI category distribution and patterns
    
    Techniques Applied:
    - Frequency distribution: counting occurrences per category
    - Percentage calculation: relative proportions
    - Category ordering: from Good to Hazardous
    - Air quality interpretation: health impact assessment
    """
    print("\n" + "="*80)
    print("4. AQI CATEGORY DISTRIBUTION")
    print("="*80)
    
    if 'aqi_category_numeric' in df.columns:
        # Map numeric to labels
        category_map = {
            0: 'Unknown', 1: 'Good', 2: 'Moderate', 
            3: 'Unhealthy for Sensitive', 4: 'Unhealthy', 
            5: 'Very Unhealthy', 6: 'Hazardous'
        }
        
        valid_df = df[df['aqi_category_numeric'] != 0]
        
        if len(valid_df) > 0:
            category_counts = valid_df['aqi_category_numeric'].value_counts().sort_index()
            
            print("\nAir Quality Distribution:")
            print("-"*80)
            
            total = len(valid_df)
            for cat_num, count in category_counts.items():
                cat_name = category_map.get(cat_num, 'Unknown')
                pct = (count / total * 100)
                bar = '█' * int(pct / 2)
                print(f"  {cat_name:25s}: {count:5d} ({pct:5.1f}%) {bar}")
            
            print("\n" + "-"*80)
            print("Health Impact Interpretation:")
            print("-"*80)
            print("  • Good (0-50): Air quality satisfactory, minimal risk")
            print("  • Moderate (51-100): Acceptable, some sensitive individuals affected")
            print("  • Unhealthy for Sensitive (101-150): Sensitive groups may experience effects")
            print("  • Unhealthy (151-200): Everyone may begin to experience effects")
            print("  • Very Unhealthy (201-300): Health alert, everyone affected")
            print("  • Hazardous (301+): Health warning, emergency conditions")


# ============================================================================
# 3. TEMPORAL PATTERNS & TRENDS
# ============================================================================

def temporal_analysis(df):
    """
    Identifies time-based patterns in AQI data
    
    Techniques Applied:
    - Time series conversion: creating proper datetime index
    - Resampling: aggregating to daily/monthly averages
    - Trend detection: identifying long-term patterns
    - Moving averages: smoothing short-term fluctuations
    """
    print("\n" + "="*80)
    print("5. TEMPORAL PATTERNS")
    print("="*80)
    
    if 'timestamp' in df.columns or 'datetime' in df.columns:
        time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        df_time = df_time.set_index(time_col)
        
        # Filter valid AQI
        df_time = df_time[(df_time['aqi'] != -1) & (df_time['aqi'].notna())]
        
        print("\nTime Range:")
        print("-"*80)
        print(f"  Start: {df_time.index.min()}")
        print(f"  End: {df_time.index.max()}")
        print(f"  Duration: {(df_time.index.max() - df_time.index.min()).days} days")
        
        # Hourly patterns
        if 'hour' in df.columns:
            print("\n" + "-"*80)
            print("Hourly AQI Patterns:")
            print("-"*80)
            
            hourly_avg = df[df['aqi'] != -1].groupby('hour')['aqi'].mean()
            
            print(f"  Peak AQI Hour: {hourly_avg.idxmax()}:00 (avg: {hourly_avg.max():.1f})")
            print(f"  Best AQI Hour: {hourly_avg.idxmin()}:00 (avg: {hourly_avg.min():.1f})")
        
        # Day of week patterns
        if 'day_of_week' in df.columns:
            print("\n" + "-"*80)
            print("Day of Week AQI Patterns:")
            print("-"*80)
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_avg = df[df['aqi'] != -1].groupby('day_of_week')['aqi'].mean()
            
            for day_num, aqi_avg in dow_avg.items():
                if day_num < len(day_names):
                    print(f"  {day_names[day_num]:10s}: {aqi_avg:.1f}")
        
        # Monthly trends
        if len(df_time) > 30:
            print("\n" + "-"*80)
            print("Monthly AQI Trends:")
            print("-"*80)
            
            monthly_avg = df_time['aqi'].resample('M').mean()
            
            if len(monthly_avg) > 0:
                print(f"  Highest Month: {monthly_avg.idxmax().strftime('%B %Y')} (avg: {monthly_avg.max():.1f})")
                print(f"  Lowest Month: {monthly_avg.idxmin().strftime('%B %Y')} (avg: {monthly_avg.min():.1f})")
                
                # Overall trend
                if len(monthly_avg) >= 3:
                    recent_avg = monthly_avg[-3:].mean()
                    earlier_avg = monthly_avg[:3].mean()
                    
                    if recent_avg > earlier_avg:
                        trend = "worsening"
                        change_pct = ((recent_avg - earlier_avg) / earlier_avg * 100)
                    else:
                        trend = "improving"
                        change_pct = ((earlier_avg - recent_avg) / earlier_avg * 100)
                    
                    print(f"\n  Overall Trend: {trend.upper()} ({change_pct:+.1f}%)")


# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(df):
    """
    Examines relationships between features
    
    Techniques Applied:
    - Pearson correlation: measuring linear relationships
    - Correlation matrix: pairwise feature relationships
    - Feature importance: identifying strong predictors
    - Multicollinearity detection: finding redundant features
    """
    print("\n" + "="*80)
    print("6. CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Focus on key features
    key_features = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                    'temperature', 'humidity', 'pressure', 'wind_speed']
    
    available_features = [c for c in key_features if c in numeric_cols]
    
    if len(available_features) > 1:
        # Filter valid data
        corr_df = df[available_features].replace(-1, np.nan)
        
        # Compute correlation matrix
        corr_matrix = corr_df.corr()
        
        print("\nTop Correlations with AQI:")
        print("-"*80)
        
        if 'aqi' in corr_matrix.columns:
            aqi_corr = corr_matrix['aqi'].sort_values(ascending=False)
            
            for feature, corr_val in aqi_corr.items():
                if feature != 'aqi':
                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak"
                    direction = "positive" if corr_val > 0 else "negative"
                    
                    print(f"  {feature:20s}: {corr_val:+.3f} ({strength} {direction})")
        
        print("\n" + "-"*80)
        print("Correlation Interpretation:")
        print("-"*80)
        print("  • |r| > 0.7: Strong correlation")
        print("  • 0.4 < |r| < 0.7: Moderate correlation")
        print("  • |r| < 0.4: Weak correlation")
        print("  • Positive: Features increase together")
        print("  • Negative: One increases as other decreases")
        
        # Multicollinearity check
        print("\n" + "-"*80)
        print("Highly Correlated Feature Pairs (potential multicollinearity):")
        print("-"*80)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.85:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("  ✓ No severe multicollinearity detected")


# ============================================================================
# 5. OUTLIER DETECTION
# ============================================================================

def outlier_detection(df):
    """
    Identifies anomalous data points
    
    Techniques Applied:
    - IQR method: Interquartile Range for robust outlier detection
    - Z-score: Standard deviations from mean
    - Boxplot rule: Values beyond 1.5 × IQR from quartiles
    - Domain knowledge: Checking against realistic AQI ranges
    """
    print("\n" + "="*80)
    print("7. OUTLIER DETECTION")
    print("="*80)
    
    key_cols = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity']
    available_cols = [c for c in key_cols if c in df.columns]
    
    print("\nIQR Method (1.5 × IQR rule):")
    print("-"*80)
    
    for col in available_cols:
        valid_data = df[df[col] != -1][col].dropna()
        
        if len(valid_data) > 0:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
            outlier_pct = (len(outliers) / len(valid_data) * 100)
            
            print(f"  {col.upper():15s}: {len(outliers):5d} outliers ({outlier_pct:.2f}%) "
                  f"[Valid range: {lower_bound:.1f} - {upper_bound:.1f}]")
    
    print("\n" + "-"*80)
    print("Outlier Interpretation:")
    print("-"*80)
    print("  • < 5% outliers: Normal distribution, few anomalies")
    print("  • 5-10% outliers: Some extreme events (pollution spikes, weather extremes)")
    print("  • > 10% outliers: Highly variable data or measurement issues")


# ============================================================================
# 6. FEATURE RELATIONSHIPS
# ============================================================================

def feature_relationships(df):
    """
    Explores how features interact with AQI
    
    Techniques Applied:
    - Groupby aggregation: Average AQI by categories
    - Conditional analysis: AQI patterns under different conditions
    - Interaction effects: Combined feature impacts
    """
    print("\n" + "="*80)
    print("8. FEATURE RELATIONSHIPS")
    print("="*80)
    
    # Weekend vs Weekday
    if 'is_weekend' in df.columns:
        print("\nWeekend vs Weekday AQI:")
        print("-"*80)
        
        valid_df = df[df['aqi'] != -1]
        weekend_aqi = valid_df[valid_df['is_weekend'] == 1]['aqi'].mean()
        weekday_aqi = valid_df[valid_df['is_weekend'] == 0]['aqi'].mean()
        
        print(f"  Weekday Average: {weekday_aqi:.1f}")
        print(f"  Weekend Average: {weekend_aqi:.1f}")
        
        diff = weekend_aqi - weekday_aqi
        print(f"  Difference: {diff:+.1f} ({'Higher' if diff > 0 else 'Lower'} on weekends)")
    
    # Time of day
    if 'time_of_day_numeric' in df.columns:
        print("\n" + "-"*80)
        print("Time of Day AQI Patterns:")
        print("-"*80)
        
        time_labels = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        valid_df = df[df['aqi'] != -1]
        
        for time_num in range(4):
            time_df = valid_df[valid_df['time_of_day_numeric'] == time_num]
            if len(time_df) > 0:
                avg_aqi = time_df['aqi'].mean()
                print(f"  {time_labels[time_num]:20s}: {avg_aqi:.1f}")


# ============================================================================
# 7. MAIN EDA EXECUTION
# ============================================================================

def run_full_eda():
    """
    Execute complete exploratory data analysis
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  EXPLORATORY DATA ANALYSIS - AQI PREDICTION PROJECT".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nFetching data from Hopsworks...")
    df = read_features(
        api_key=HOPSWORKS_API_KEY,
        project_name=HOPSWORKS_PROJECT,
        feature_group_name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )
    
    if df is None or len(df) == 0:
        print("✗ No data found!")
        return
    
    print(f"✓ Loaded {len(df)} rows")
    
    # Run all EDA sections
    data_overview(df)
    data_quality_assessment(df)
    statistical_summary(df)
    distribution_analysis(df)
    temporal_analysis(df)
    correlation_analysis(df)
    outlier_detection(df)
    feature_relationships(df)
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  EDA COMPLETE - Ready for Model Training!".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\n📊 Key Takeaways:")
    print("-"*80)
    print("1. Data Quality: Check the quality score - aim for >95% usable data")
    print("2. Distributions: Look for skewed features that may need transformation")
    print("3. Correlations: Identify strong predictors for feature selection")
    print("4. Temporal Patterns: Understand time-based trends for better predictions")
    print("5. Outliers: Decide whether to keep/remove extreme values")
    print("\n💡 Next Steps:")
    print("-"*80)
    print("1. Update config.py: Set PREDICTION_HORIZON_DAYS = 3")
    print("2. Run: python train_and_register_model.py")
    print("3. Build web application with insights from this EDA")
    print()


if __name__ == "__main__":
    run_full_eda()
