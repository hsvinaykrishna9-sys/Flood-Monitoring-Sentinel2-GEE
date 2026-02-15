
import pandas as pd
import os
import glob

# Path to raw data
data_path = "data/raw/"

# Get all satellite yearly files
sat_files = glob.glob(os.path.join(data_path, "Kodagu_Flood_Features_*.csv"))

print("Found files:")
for f in sat_files:
    print(f)

# Combine all years
df_list = [pd.read_csv(file) for file in sat_files]
satellite_all = pd.concat(df_list, ignore_index=True)

# Create year_month column
satellite_all["year_month"] = (
    satellite_all["year"].astype(str) + "-" +
    satellite_all["month"].astype(str).str.zfill(2)
)

# Sort properly
satellite_all = satellite_all.sort_values(["year", "month"])

# Save combined file
os.makedirs("data/processed", exist_ok=True)
satellite_all.to_csv("data/processed/kodagu_satellite_monthly_2018_2025.csv", index=False)

print("\n✅ Combined satellite file saved:")
print("data/processed/kodagu_satellite_monthly_2018_2025.csv")

print("\nShape:", satellite_all.shape)
print("\nSample:")
print(satellite_all.head())
