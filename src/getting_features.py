from config import DATASET_DIR, TRACKS, TELEMETRY_FILES, RACE
import pandas as pd 
import numpy as np 
import re
print(DATASET_DIR)
print((DATASET_DIR) / TRACKS[0] )
barber = pd.read_csv((DATASET_DIR) / TRACKS[0]/ RACE / TELEMETRY_FILES[0] )
# sonoma = pd.read_csv(DATASET_DIR / TRACKS[1]/RACE / TELEMETRY_FILES[1] )
# vir = pd.read_csv(DATASET_DIR / TRACKS[2] /RACE /TELEMETRY_FILES[2] )
print(barber.head())
# print(sonoma.head())
# print(vir.head())