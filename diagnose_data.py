"""
Diagnostic script to check why data isn't being stored properly
"""
from feature_store import read_features
import os
import pandas as pd

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")

print("="*60)
print("HOPSWORKS DATA DIAGNOSTICS")
print("="*60)

df = read_features(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)

if df is not None:
    print(f"\n✓ Total rows in Hopsworks: {len(df)}")
    print(f"✓ Unique timestamps: {df['timestamp'].nunique()}")
    print(f"✓ Duplicate timestamps: {len(df) - df['timestamp'].nunique()}")
    
    # Check time range
    df['datetime'] = pd.to_datetime(df['timestamp'])
    print(f"\n📅 Time Range:")
    print(f"   First: {df['datetime'].min()}")
    print(f"   Last:  {df['datetime'].max()}")
    print(f"   Span:  {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    # Check for gaps
    df_sorted = df.sort_values('datetime')
    df_sorted['time_diff'] = df_sorted['datetime'].diff()
    
    print(f"\n⏱️  Time Gaps:")
    print(f"   Min gap: {df_sorted['time_diff'].min()}")
    print(f"   Max gap: {df_sorted['time_diff'].max()}")
    print(f"   Mean gap: {df_sorted['time_diff'].mean()}")
    
    # Show recent entries
    print(f"\n📊 Last 10 Timestamps:")
    recent = df_sorted.tail(10)[['datetime', 'aqi', 'pm25', 'timestamp']].copy()
    recent['hour_gap'] = recent['datetime'].diff().dt.total_seconds() / 3600
    for idx, row in recent.iterrows():
        gap_str = f"(+{row['hour_gap']:.1f}h)" if pd.notna(row['hour_gap']) else ""
        print(f"   {row['datetime']} | AQI: {row['aqi']:>3} | PM2.5: {row['pm25']:>5} {gap_str}")
    
    # Check if primary key violations
    if len(df) > df['timestamp'].nunique():
        print(f"\n⚠️  WARNING: Duplicate primary keys detected!")
        duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
        print(f"   Duplicate rows: {len(duplicates)}")
        print(f"   Affected timestamps:")
        for ts in duplicates['timestamp'].unique():
            print(f"      {ts}")
    
    # Estimate expected rows
    hours_elapsed = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
    expected_rows = int(hours_elapsed)
    print(f"\n🎯 Expected vs Actual:")
    print(f"   Time span: {hours_elapsed:.1f} hours")
    print(f"   Expected hourly rows: ~{expected_rows}")
    print(f"   Actual rows: {len(df)}")
    print(f"   Collection rate: {len(df)/hours_elapsed*100:.1f}% of expected")
    
else:
    print("✗ Failed to read data from Hopsworks")

print("\n" + "="*60)
