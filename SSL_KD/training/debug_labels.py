import sys
sys.path.insert(0, r'D:\ECG_SSL_KD')
from SSL_KD.datasets.cinc2020_dataset import load_cinc_header
import pandas as pd
import os

data_dir = r'D:\Datasets\phsionet_2020'
mapping_path = os.path.join(data_dir, 'dx_mapping_scored.csv')
dx_df = pd.read_csv(mapping_path)
scored_classes = dx_df['SNOMED CT Code'].astype(str).tolist()

EQUIVALENT_CLASSES = {
    "59118001": "713427006",
    "63593006": "284470004",
    "17338001": "427172004"
}

mapped_classes = set()
for code in scored_classes:
    mapped_classes.add(EQUIVALENT_CLASSES.get(code, code))

target_classes = sorted(list(mapped_classes))
class_to_idx = {str(code): i for i, code in enumerate(target_classes)}

print("class_to_idx keys:", list(class_to_idx.keys())[:5])

fs, diagnoses = load_cinc_header(r'D:/Datasets/phsionet_2020\training\cpsc_2018\g1\A0002.hea')
print("diagnoses from load_cinc_header:", diagnoses)

for dx in diagnoses:
    print(f"Checking '{dx}'")
    print(f"type: {type(dx)}")
    mapped_dx = EQUIVALENT_CLASSES.get(dx, dx)
    print(f"mapped_dx: '{mapped_dx}', in class_to_idx? {mapped_dx in class_to_idx}")
