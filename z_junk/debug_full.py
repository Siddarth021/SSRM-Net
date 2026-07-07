import traceback
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("Config loading...")
with open("ECG_SSL_KD/configs/finetune.yaml") as f:
    config = yaml.safe_load(f)
print(f"  data_dir = {config['data']['data_dir']}")
print(f"  score_path = {config['data']['score_path']}")
print(f"  ssl_path = {config['checkpoints']['pretrained_ssl_teacher']}")
print(f"  mode = {config['training']['mode']}")

print("Model creation...")
from ECG_SSL_KD.training.finetune_teacher import TeacherFineTuneModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device = {device}")
model = TeacherFineTuneModel(num_classes=5, mode="frozen").to(device)
print("  Model created OK")

print("Loading SSL weights...")
ssl_path = config["checkpoints"]["pretrained_ssl_teacher"]
if os.path.exists(ssl_path):
    model.encoder.load_state_dict(torch.load(ssl_path, weights_only=False))
    print(f"  Loaded from {ssl_path}")
else:
    print(f"  WARNING: {ssl_path} not found")

print("Setting up run dir...")
from datetime import datetime
import shutil
save_dir = config["checkpoints"]["save_dir"]
run_dir = os.path.join(save_dir, "debug_test_run")
os.makedirs(run_dir, exist_ok=True)
log_path = os.path.join(run_dir, "train.log")

def log_print(msg):
    print(msg)
    with open(log_path, "a") as lf:
        lf.write(msg + "\n")

shutil.copy("ECG_SSL_KD/configs/finetune.yaml", os.path.join(run_dir, "config.yaml"))

log_print("Loading PTB-XL fine-tuning dataset...")
from ECG_SSL_KD.datasets.ptbxl_dataset import PTBXLECG
ecg_dataset = PTBXLECG(config["data"]["data_dir"], config["data"]["score_path"], config["data"]["split"])
train_ds, val_ds, test_ds = ecg_dataset.data_prepare()

sample = train_ds[0]
log_print(f"Train samples: {len(train_ds)}")
log_print(f"Validation samples: {len(val_ds)}")
log_print(f"lead2 shape: {sample['lead2'].shape}")
log_print(f"morphology shape: {sample['morphology'].shape}")
log_print(f"label shape: {sample['label'].shape}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
log_print(f"train_loader batches: {len(train_loader)}")
log_print(f"val_loader batches: {len(val_loader)}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

log_print("Entering training loop (2 epochs test)...")
import time
for epoch in range(1, 3):
    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        l2 = batch["lead2"].to(device)
        mo = batch["morphology"].to(device)
        la = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(l2, mo)
        loss = criterion(logits, la)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * l2.size(0)
    epoch_loss /= len(train_loader.dataset)
    log_print(f"epoch={epoch} | loss={epoch_loss:.4f} | time={time.time()-t0:.1f}s")

log_print("TRAINING COMPLETED SUCCESSFULLY")
