"""
generate_dataset.py
--------------------
Creates a synthetic-but-physically-grounded dataset for the AC Capacity
Prediction project. AC tonnage is derived from a simplified cooling-load
model (room volume, occupancy heat gain, solar/window gain, equipment
load, outdoor temperature delta, and insulation quality) plus realistic
sensor/measurement noise -- so the ML model has genuine signal to learn,
not pure randomness.

Run:  python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# ---- Feature generation -------------------------------------------------
room_area           = np.random.uniform(9, 45, N)          # m^2
room_height         = np.random.uniform(2.4, 3.6, N)       # m
occupancy           = np.random.randint(1, 11, N)           # people
outdoor_temperature = np.random.uniform(24, 45, N)          # deg C
window_area         = np.random.uniform(1, 12, N)           # m^2
equipment_load      = np.random.uniform(0.1, 3.5, N)        # kW
insulation_level    = np.random.randint(1, 6, N)             # 1 (poor) - 5 (excellent)
sun_exposure        = np.random.randint(1, 4, N)             # 1 low, 2 medium, 3 high

# ---- Ground-truth cooling load model ------------------------------------
# Rule-of-thumb HVAC estimation, scaled into "tons" of refrigeration.
volume_load     = (room_area * room_height) * 0.018
occupancy_load  = occupancy * 0.08
temp_load       = np.clip(outdoor_temperature - 24, 0, None) * 0.02
window_load     = window_area * sun_exposure * 0.025
equipment_gain  = equipment_load * 0.25
insulation_relief = insulation_level * 0.10

ac_capacity = (
    volume_load
    + occupancy_load
    + temp_load
    + window_load
    + equipment_gain
    - insulation_relief
    + 0.35                      # base fixed load
)

# realistic measurement/estimation noise
noise = np.random.normal(0, 0.15, N)
ac_capacity = ac_capacity + noise
ac_capacity = np.clip(ac_capacity, 0.75, 5.5)

df = pd.DataFrame({
    "Room_Area": room_area.round(2),
    "Room_Height": room_height.round(2),
    "Occupancy": occupancy,
    "Outdoor_Temperature": outdoor_temperature.round(1),
    "Window_Area": window_area.round(2),
    "Equipment_Load": equipment_load.round(2),
    "Insulation_Level": insulation_level,
    "Sun_Exposure": sun_exposure,
    "AC_Capacity_Ton": ac_capacity.round(2),
})

df.to_csv("data/AC_Capacity_Dataset_2000.csv", index=False)
print(f"Dataset created: data/AC_Capacity_Dataset_2000.csv  ({len(df)} rows)")
print(df.describe().T)
