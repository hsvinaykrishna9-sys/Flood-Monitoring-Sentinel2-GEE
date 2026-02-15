
import requests
import pandas as pd
import os


class WeatherCollector:
    """
    Collect historical weather data for Kodagu
    using Open-Meteo API
    """

    def __init__(self, latitude=12.4244, longitude=75.7382, output_dir="data/raw"):
        self.latitude = latitude
        self.longitude = longitude
        self.api_url = "https://archive-api.open-meteo.com/v1/archive"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fetch_daily(self, start_date, end_date):

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "Asia/Kolkata"
        }

        print("🌦 Fetching weather data...")
        print(f"📅 Period: {start_date} to {end_date}")

        response = requests.get(self.api_url, params=params)
        response.raise_for_status()

        data = response.json()["daily"]

        df = pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temperature_c": data["temperature_2m_mean"],
            "rainfall_mm": data["precipitation_sum"],
            "humidity_percent": data["relative_humidity_2m_mean"]
        })

        return df

    def aggregate_monthly(self, df_daily):

        df_daily = df_daily.set_index("date")

        # IMPORTANT: Use 'ME' for Pandas 2.2+
        monthly = pd.DataFrame({
            "temperature_c": df_daily["temperature_c"].resample("ME").mean(),
            "rainfall_mm": df_daily["rainfall_mm"].resample("ME").sum(),
            "humidity_percent": df_daily["humidity_percent"].resample("ME").mean()
        })

        monthly = monthly.round(2)

        # Convert index to YYYY-MM format
        monthly.index = monthly.index.strftime("%Y-%m")

        monthly = monthly.reset_index()
        monthly = monthly.rename(columns={"index": "year_month"})

        return monthly

    def run(self,
            start_date="2018-01-01",
            end_date="2025-12-31"):

        daily = self.fetch_daily(start_date, end_date)
        monthly = self.aggregate_monthly(daily)

        daily_path = os.path.join(self.output_dir, "kodagu_weather_daily_2018_2025.csv")
        monthly_path = os.path.join(self.output_dir, "kodagu_weather_monthly_2018_2025.csv")

        daily.to_csv(daily_path, index=False)
        monthly.to_csv(monthly_path, index=False)

        print("\n✅ Files saved:")
        print("   ", daily_path)
        print("   ", monthly_path)

        return daily, monthly


if __name__ == "__main__":

    collector = WeatherCollector()
    daily_df, monthly_df = collector.run()

    print("\n📊 Monthly Sample:")
    print(monthly_df.head())

    print("\n📈 Total Months:", len(monthly_df))
