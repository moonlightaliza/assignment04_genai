import os, sys, json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from starter.uhat import load_uhat_characters
from starter.viz import save_grid
from model import VariationalAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = os.path.join(ROOT, "results")
os.makedirs(res, exist_ok=True)
torch.manual_seed(0)

model = VariationalAutoEncoder().to(device)
model.load_state_dict(torch.load("vae.pth", map_location=device))
model.eval()

data_root = os.path.join(ROOT, "starter", "data", "uhat")
x_tr, _, names = load_uhat_characters(root=data_root, split="train")
x_te, l_te, _ = load_uhat_characters(root=data_root, split="test", class_names=names)

with torch.no_grad():
    # (a) 16 test images and their reconstructions (row 1 originals, row 2 reconstructions)
    x = x_te[:16].to(device)
    logits, _, _ = model(x)
    recon = torch.sigmoid(logits)
    save_grid(torch.cat([x.cpu(), recon.cpu()]), os.path.join(res, "vae_reconstructions.png"), nrow=16)

    # (b) 64 new characters from z ~ N(0, I)
    z = torch.randn(64, 16, device=device)
    samples = torch.sigmoid(model.decode(z))
    save_grid(samples.cpu(), os.path.join(res, "vae_samples.png"), nrow=8)

    # (c) interpolation between mu of two test images of DIFFERENT characters
    i = 0
    j = int((l_te != l_te[i]).nonzero()[0])          # first test image with a different label
    mu, _ = model.encode(x_te[[i, j]].to(device))    # (2, 16)
    t = torch.linspace(0, 1, 10, device=device).unsqueeze(1)   # (10, 1)
    z_line = (1 - t) * mu[0] + t * mu[1]             # (10, 16), equally spaced on the line
    interp = torch.sigmoid(model.decode(z_line))
    save_grid(interp.cpu(), os.path.join(res, "vae_interpolation.png"), nrow=10,
              titles=[names[l_te[i]]] + [""] * 8 + [names[l_te[j]]])

# loss curves
log = json.load(open(os.path.join(res, "vae_log.json")))
ep = range(1, len(log["train"]) + 1)
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, k, title in zip(axs, [0, 1, None], ["Reconstruction", "KL", "Negative ELBO (sum)"]):
    for split, style in [("train", "-"), ("test", "--")]:
        vals = [v[k] if k is not None else v[0] + v[1] for v in log[split]]
        ax.plot(ep, vals, style, label=split)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("nats / image")
    ax.grid(True); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(res, "vae_loss_terms.png"), dpi=200)
print("saved figures to results/")