import os, sys, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from starter.uhat import load_uhat_characters
from starter.viz import save_grid
from schedule import make_schedule, T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

unet = UNet2DModel(...).to(device)     # SAME constructor as train.py
unet.load_state_dict(torch.load("diffusion.pth", map_location=device))
unet.eval()
betas, alphas, alpha_bars = make_schedule(device)

# class names (same order as training)
data_root = os.path.join(ROOT, "starter", "data", "uhat")
_, _, names = load_uhat_characters(root=data_root, split="train")

@torch.no_grad()
def sample(y, keep=()):
    """y: (N,) class labels. Returns final images in [-1,1], plus {t: x_t} for t in keep."""
    x = torch.randn(len(y), 1, 28, 28, device=device)          # x_T ~ N(0, I)
    saved = {}
    for t in reversed(range(T)):                                # 999 ... 0
        if t in keep:
            saved[t] = x.clone()
        tt = torch.full((len(y),), t, device=device, dtype=torch.long)
        eps_pred = unet(x, tt, class_labels=y).sample

        alpha_t = alphas[t]
        abar_t = alpha_bars[t]
        beta_t = betas[t]

        mean = (x - (1 - alpha_t) / (1 - abar_t).sqrt() * eps_pred) / alpha_t.sqrt()

        if t > 0:
            z = torch.randn_like(x)
            x = mean + beta_t.sqrt() * z
        else:
            x = mean
    return x, saved

# C.5a: 8 samples for each of the 40 characters -> 40 x 8 grid, one row per character
y = torch.arange(40, device=device).repeat_interleave(8)      # 0,0,..,0,1,1,..
imgs, _ = sample(y)
imgs = (imgs + 1) / 2                                          # back to [0,1]
fig, axes = plt.subplots(40, 8, figsize=(8, 40))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i, 0].cpu(), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    if i % 8 == 0:
        ax.set_ylabel(names[i // 8], rotation=0, ha="right", va="center", fontsize=7)
fig.savefig(os.path.join(ROOT, "results", "diffusion_samples.png"), dpi=150, bbox_inches="tight")

# C.5b: denoising of one sample at t in {999,750,500,250,100,0}
ts = [999, 750, 500, 250, 100, 0]
_, saved = sample(torch.tensor([0], device=device), keep=set(ts))
row = torch.cat([(saved[t].clamp(-1, 1) + 1) / 2 for t in ts])
save_grid(row, os.path.join(ROOT, "results", "diffusion_denoising.png"),
          nrow=6, titles=[f"t={t}" for t in ts])