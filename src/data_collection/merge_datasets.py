
import pandas as pd
import os

sat_path = "data/processed/kodagu_satellite_monthly_2018_2025.csv"
weather_path = "data/raw/kodagu_weather_monthly_2018_2025.csv"

sat = pd.read_csv(sat_path)
weather = pd.read_csv(weather_path)

# Ensure correct column name
weather = weather.rename(columns={"date": "year_month"})

# Merge
merged = pd.merge(sat, weather, on="year_month", how="inner")

# Save final dataset
os.makedirs("data/processed", exist_ok=True)
merged.to_csv("data/processed/kodagu_final_dataset_2018_2025.csv", index=False)

print("\n✅ Final merged dataset created")
print("Shape:", merged.shape)
print("\nSample:")
print(merged.head())
