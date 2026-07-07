import traceback
import sys

print("Step 1: Importing modules...")
try:
    from ECG_SSL_KD.datasets.ptbxl_dataset import PTBXLECG
    print("Step 1: OK - Imported PTBXLECG")
except Exception as e:
    print("Step 1 FAILED - Could not import PTBXLECG:")
    traceback.print_exc()
    sys.exit(1)

print("Step 2: Creating PTBXLECG instance...")
try:
    ecg_dataset = PTBXLECG("D:\\Datasets\\PTBXL\\")
    print("Step 2: OK - PTBXLECG instance created")
except Exception as e:
    print("Step 2 FAILED - PTBXLECG constructor raised:")
    traceback.print_exc()
    sys.exit(1)

print("Step 3: Calling data_prepare()...")
try:
    train_ds, val_ds, test_ds = ecg_dataset.data_prepare()
    print(f"Step 3: OK - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
except Exception as e:
    print("Step 3 FAILED - data_prepare() raised:")
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Verifying dataset sizes...")
if len(train_ds) == 0:
    print("FATAL: train_ds is EMPTY")
    sys.exit(1)
if len(val_ds) == 0:
    print("FATAL: val_ds is EMPTY")
    sys.exit(1)
print(f"Step 4: OK - train={len(train_ds)}, val={len(val_ds)}")

print("Step 5: Fetching first sample from train_ds...")
try:
    sample = train_ds[0]
    print(f"Step 5: OK - keys={list(sample.keys())}")
    print(f"  lead2.shape = {sample['lead2'].shape}")
    print(f"  morphology.shape = {sample['morphology'].shape}")
    print(f"  label.shape = {sample['label'].shape}")
    print(f"  label values = {sample['label']}")
except Exception as e:
    print("Step 5 FAILED - Could not get first sample:")
    traceback.print_exc()
    sys.exit(1)

print("\nALL STEPS PASSED - Dataset is healthy.")
