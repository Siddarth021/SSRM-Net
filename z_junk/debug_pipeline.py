import traceback
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("Step 1: Loading config...")
try:
    with open("ECG_SSL_KD/configs/finetune.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"  data_dir = {config['data']['data_dir']}")
    print(f"  pretrained_ssl_teacher = {config['checkpoints']['pretrained_ssl_teacher']}")
    print(f"  mode = {config['training']['mode']}")
    print(f"  num_classes = {config['training']['num_classes']}")
    print(f"  epochs = {config['training']['epochs']}")
    print(f"  batch_size = {config['training']['batch_size']}")
    print(f"  device = {config['training']['device']}")
    print("Step 1: OK")
except Exception as e:
    print("Step 1 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Creating model...")
try:
    from ECG_SSL_KD.training.finetune_teacher import TeacherFineTuneModel
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model = TeacherFineTuneModel(num_classes=config['training']['num_classes'], mode=config['training']['mode']).to(device)
    print("Step 2: OK - model created")
except Exception as e:
    print("Step 2 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Loading SSL encoder weights...")
try:
    ssl_path = config['checkpoints']['pretrained_ssl_teacher']
    if os.path.exists(ssl_path):
        model.encoder.load_state_dict(torch.load(ssl_path, weights_only=False))
        print(f"Step 3: OK - weights loaded from {ssl_path}")
    else:
        print(f"Step 3: WARNING - SSL weights not found at {ssl_path}")
except Exception as e:
    print("Step 3 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Loading dataset...")
try:
    from ECG_SSL_KD.datasets.ptbxl_dataset import PTBXLECG
    ecg_dataset = PTBXLECG(config['data']['data_dir'], config['data']['score_path'], config['data']['split'])
    train_ds, val_ds, test_ds = ecg_dataset.data_prepare()
    print(f"Step 4: OK - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
except Exception as e:
    print("Step 4 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Creating DataLoaders...")
try:
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    print(f"Step 5: OK - train_loader batches={len(train_loader)}, val_loader batches={len(val_loader)}")
except Exception as e:
    print("Step 5 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Creating optimizer and loss...")
try:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    print("Step 6: OK")
except Exception as e:
    print("Step 6 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 7: Running 1 training batch...")
try:
    model.train()
    batch = next(iter(train_loader))
    lead2 = batch["lead2"].to(device)
    morphology = batch["morphology"].to(device)
    labels = batch["label"].to(device)
    print(f"  lead2={lead2.shape}, morphology={morphology.shape}, labels={labels.shape}")
    optimizer.zero_grad()
    logits = model(lead2, morphology)
    print(f"  logits={logits.shape}")
    loss = criterion(logits, labels)
    print(f"  loss={loss.item():.4f}")
    loss.backward()
    optimizer.step()
    print("Step 7: OK - Training batch completed!")
except Exception as e:
    print("Step 7 FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL STEPS PASSED - Training pipeline is healthy ===")
