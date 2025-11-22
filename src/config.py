import os 
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR_2 = BASE_DIR / "dataset" / 'archive' / 'UCI-HAR Dataset'
TRACKS = [
        "barber",
        "Sonoma",
        "VIR"
    ]
RACE = 'RACE 1'
TELEMETRY_FILES =  [
    "R1_barber_telemetry_data.csv",
    "sonoma_telemetry_R1.csv",
    "R1_vir_telemetry_data.csv"
]
if __name__ == "__main__":
    print(BASE_DIR)
    print(DATASET_DIR)
    print(os.listdir(DATASET_DIR))

    for track in os.listdir(DATASET_DIR):
        print(track)
        print(os.listdir(DATASET_DIR / track / 'Race 1'))
        print('\n')
    
