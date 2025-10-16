from data_fetcher import fetch_daily_data
from feature_store import store_features
from datetime import datetime


WAQI_API_TOKEN = "088661c637816f9f1463ca3e44d37da6d739d021"
STATION_ID = "A401143"
HOPSWORKS_API_KEY = "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5"
HOPSWORKS_PROJECT = "AQI_Project_10"


def main():
    print(f"=== AQI Daily Update: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    
    df = fetch_daily_data(WAQI_API_TOKEN, STATION_ID)
    
    if df is not None:
        print(f"Collected: 1 sample with {len(df.columns)} features")
        
        success = store_features(
            df=df,
            api_key=HOPSWORKS_API_KEY,
            project_name=HOPSWORKS_PROJECT,
            feature_group_name="aqi_features",
            version=1
        )
        
        if success:
            print("✓ Daily update completed")
        else:
            print("✗ Daily update failed")
            df.to_csv(f"backup_daily_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
    else:
        print("✗ No data collected")


if __name__ == "__main__":
    main()
