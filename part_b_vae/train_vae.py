import os, sys, json, time
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from starter.uhat import load_uhat_characters
from model import VariationalAutoEncoder, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_root = os.path.join(ROOT, "starter", "data", "uhat")
x_tr, l_tr, names = load_uhat_characters(root=data_root, split="train")
x_te, _, _ = load_uhat_characters(root=data_root, split="test", class_names=names)
print(x_tr.shape, x_te.shape, len(names))      # check: 40 classes

train_loader = DataLoader(TensorDataset(x_tr), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(x_te), batch_size=128)

model = VariationalAutoEncoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def run_epoch(loader, train):
    model.train(train)
    rec_sum = kl_sum = 0.0
    n = 0
    for (x,) in loader:
        x = x.to(device)
        with torch.set_grad_enabled(train):
            logits, mu, logvar = model(x)
            recon, kl = vae_loss(logits, x, mu, logvar)
            loss = recon + kl
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        rec_sum += recon.item() * x.size(0)
        kl_sum += kl.item() * x.size(0)
        n += x.size(0)
    return rec_sum / n, kl_sum / n

log = {"train": [], "test": []}
start = time.time()
for epoch in range(1, 31):
    tr = run_epoch(train_loader, True)
    te = run_epoch(test_loader, False)
    log["train"].append(tr); log["test"].append(te)
    print(f"Epoch {epoch:2d} | train recon {tr[0]:.1f} kl {tr[1]:.1f} sum {sum(tr):.1f}"
          f" | test recon {te[0]:.1f} kl {te[1]:.1f} sum {sum(te):.1f}")

log["params"] = sum(p.numel() for p in model.parameters())
log["train_time_sec"] = time.time() - start
torch.save(model.state_dict(), "vae.pth")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "vae_log.json"), "w") as f:
    json.dump(log, f)