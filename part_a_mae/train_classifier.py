import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from starter.eurosat import (load_eurosat_split, few_label_indices,
                             patchify, CLASS_NAMES)
from model import MaskedAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ---- load the pretrained MAE and freeze it ----
model = MaskedAutoEncoder().to(device)
model.load_state_dict(torch.load("mae_pretrained.pth", map_location=device))
model.eval()                                   # turns dropout off
for p in model.parameters():
    p.requires_grad = False


@torch.no_grad()
def extract_features(images, bs=256):
    """images (N,3,64,64) -> features (N,192): all 64 patches, no masking, mean over patches."""
    feats = []
    for i in range(0, len(images), bs):
        patches = patchify(images[i:i + bs]).to(device)
        x, _, _ = model.forward_encoder(patches, apply_mask=False)   # (b, 64, 192)
        feats.append(x.mean(dim=1).cpu())                            # (b, 192)
    return torch.cat(feats)


# ---- data ----
x_tr, y_tr = load_eurosat_split("train")
x_te, y_te = load_eurosat_split("test")

few = few_label_indices(y_tr)                  # 160 fixed images, 16 per class
f_train = extract_features(x_tr[few]).to(device)     # (160, 192)
y_train = y_tr[few].to(device)
f_test = extract_features(x_te).to(device)           # (5400, 192)
y_test = y_te.to(device)
print("train features:", tuple(f_train.shape), "test features:", tuple(f_test.shape))

# ---- linear classifier: AdamW, lr 1e-3, 200 epochs, all 160 images in one batch ----
clf = nn.Linear(192, 10).to(device)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = loss_fn(clf(f_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | loss {loss.item():.4f}")

# ---- evaluation ----
with torch.no_grad():
    pred = clf(f_test).argmax(dim=1)
acc = (pred == y_test).float().mean().item()
print(f"\nTest accuracy: {acc * 100:.2f}%  (chance = 10%)")

# 10x10 confusion matrix (rows = true, cols = predicted)
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_test.cpu().numpy(), pred.cpu().numpy()):
    cm[t, p] += 1

# most confused pairs, for analysis E.2
pairs = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
         for i in range(10) for j in range(10) if i != j]
print("\nMost confused (count, true -> predicted):")
for c, a, b in sorted(pairs, reverse=True)[:6]:
    print(f"  {c:4d}  {a} -> {b}")

# ---- save outputs ----
res_dir = os.path.join(ROOT, "results")
os.makedirs(res_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"MAE few-label classifier (test acc {acc * 100:.1f}%)")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "mae_confusion_matrix.png"), dpi=200)
plt.close(fig)

with open(os.path.join(res_dir, "mae_classifier_results.json"), "w") as f:
    json.dump({"test_accuracy": acc, "confusion_matrix": cm.tolist()}, f)
print("\nSaved confusion matrix and results to results/")