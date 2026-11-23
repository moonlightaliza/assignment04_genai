import os, sys, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "starter"))   # viz.py imports nothing from starter, but this keeps things simple

from starter.eurosat import load_eurosat_split, patchify, unpatchify
from starter.viz import save_grid
from model import MaskedAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedAutoEncoder().to(device)
model.load_state_dict(torch.load("mae_pretrained.pth", map_location=device))
model.eval()

x_te, _ = load_eurosat_split("test")
imgs = x_te[:8]
patches = patchify(imgs).to(device)                 # (8, 64, 192)

with torch.no_grad():
    _, pred, mask = model(patches)                  # mask: 1 = masked

m = mask.unsqueeze(-1)                              # (8, 64, 1)
grey = torch.full_like(patches, 0.5)
masked_input = patches * (1 - m) + grey * m         # masked patches grey
recon = patches * (1 - m) + pred * m                # predicted patches pasted into visible ones

rows = [unpatchify(t.cpu()) for t in (masked_input, recon, patches)]
grid = torch.cat(rows, dim=0)                       # (24, 3, 64, 64): row 1, row 2, row 3
os.makedirs("../results", exist_ok=True)
save_grid(grid, "../results/mae_reconstructions.png", nrow=8)