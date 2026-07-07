import os
import wfdb

dl_dir = "D:\\Datasets\\MITBIH\\"
os.makedirs(dl_dir, exist_ok=True)

print(f"Downloading MIT-BIH Arrhythmia Database to {dl_dir} ...")
wfdb.dl_database('mitdb', dl_dir)

# Verify
dat_files = [f for f in os.listdir(dl_dir) if f.endswith('.dat')]
hea_files = [f for f in os.listdir(dl_dir) if f.endswith('.hea')]
atr_files = [f for f in os.listdir(dl_dir) if f.endswith('.atr')]

print(f"Downloaded {len(dat_files)} .dat files")
print(f"Downloaded {len(hea_files)} .hea files")
print(f"Downloaded {len(atr_files)} .atr files")
