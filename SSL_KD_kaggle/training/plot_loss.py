import re
import matplotlib.pyplot as plt
import os

log_path = r"/kaggle/working/pipeline_monitor.log"
out_path = r"/kaggle/working/ssl_loss_curve.png"

epochs = []
losses = []

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r'epoch=(\d+)\s+\|\s+ssl_loss=([\d\.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', markersize=4)
plt.title('Phase 0 (SSL Pre-training) Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('NT-Xent Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path)
print(f"Plot saved to {out_path}")
