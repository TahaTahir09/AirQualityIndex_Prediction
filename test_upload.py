"""
Test Upload - Upload 10 rows to verify transformation works correctly
Run this before uploading the full 17,522 rows
"""
import pandas as pd
import sys
import os

# Import the transformation function
sys.path.append(os.path.dirname(__file__))
from upload_historical_data import transform_to_hopsworks_schema
from feature_store import store_features


def test_upload():
    print("="*70)
    print("TEST UPLOAD - Verifying transformation and upload")
    print("="*70)
    
    # Load only first 10 rows
    print("\n1. Loading first 10 rows from historical_data.csv...")
    df = pd.read_csv('historical_data.csv', nrows=10)
    print(f"   ✓ Loaded {len(df)} rows")
    
    # Transform
    print("\n2. Transforming to Hopsworks schema...")
    transformed = transform_to_hopsworks_schema(df)
    print(f"   ✓ Transformed to {len(transformed.columns)} columns")
    
    # Display sample
    print("\n3. Sample of transformed data:")
    print("-"*70)
    display_cols = ['timestamp', 'aqi', 'pm25', 'temperature', 'humidity', 
                    'hour', 'aqi_category_numeric', 'temp_humidity_interaction']
    print(transformed[display_cols].head(3))
    
    # Check data types
    print("\n4. Checking data types...")
    int32_cols = transformed.select_dtypes(include=['int32']).columns.tolist()
    int64_cols = transformed.select_dtypes(include=['int64']).columns.tolist()
    float_cols = transformed.select_dtypes(include=['float64', 'float32']).columns.tolist()
    
    print(f"   • int32 columns ({len(int32_cols)}): {int32_cols[:5]}...")
    print(f"   • int64 columns ({len(int64_cols)}): {int64_cols}")
    print(f"   • float columns ({len(float_cols)}): {float_cols[:5]}...")
    
    # Confirm upload
    print("\n" + "="*70)
    print("Ready to upload 10 test rows to Hopsworks")
    print("="*70)
    
    response = input("\nProceed with test upload? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n5. Uploading to Hopsworks...")
        
        api_key = os.getenv("HOPSWORKS_API_KEY", "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5")
        project = os.getenv("HOPSWORKS_PROJECT", "AQI_Project_10")
        
        try:
            success = store_features(
                df=transformed,
                api_key=api_key,
                project_name=project,
                feature_group_name="aqi_features",
                version=1
            )
            
            if success:
                print("\n" + "="*70)
                print("✓ TEST UPLOAD SUCCESSFUL!")
                print("="*70)
                print("\nNext steps:")
                print("1. Run 'python check_data_status.py' to verify")
                print("2. If all looks good, run 'python upload_historical_data.py'")
                print("   to upload all 17,522 rows")
            else:
                print("\n✗ Upload failed. Check error messages above.")
                
        except Exception as e:
            print(f"\n✗ Error during upload: {e}")
            print("\nTroubleshooting:")
            print("1. Check HOPSWORKS_API_KEY and HOPSWORKS_PROJECT")
            print("2. Verify network connection")
            print("3. Check Hopsworks Feature Group schema")
    else:
        print("\nTest upload cancelled.")
        print("\nTo proceed anyway:")
        print("  python upload_historical_data.py")


if __name__ == "__main__":
    test_upload()
