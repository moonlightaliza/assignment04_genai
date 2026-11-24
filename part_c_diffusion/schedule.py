import torch

T = 1000

def make_schedule(device="cpu"):
    betas = torch.linspace(1e-4, 0.02, T, device = device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def q_sample(x0, t, eps, alpha_bars):
    abar = alpha_bars[t].view(-1, 1, 1, 1)  # (B,) -> (B,1,1,1)

    return abar.sqrt() * x0 + (1 - abar).sqrt() * eps

if __name__ == "__main__":
    import os, sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(ROOT)
    from starter.uhat import load_uhat_characters
    from starter.viz import save_grid

    betas, alphas, alpha_bars = make_schedule()
    print("shapes:", betas.shape, alphas.shape, alpha_bars.shape)
    print("abar_0:", alpha_bars[0].item())
    print("abar_T:", alpha_bars[-1].item())          # expect ~4e-5

    data_root = os.path.join(ROOT, "starter", "data", "uhat")
    x, _, _ = load_uhat_characters(root=data_root, split="train")
    x0 = (x[0:1] * 2 - 1)                            # [0,1] -> [-1,1], shape (1,1,28,28)

    ts = [0, 100, 250, 500, 750, 999]
    eps = torch.randn_like(x0)                       # SAME noise for all six
    imgs = [q_sample(x0, torch.tensor([t]), eps, alpha_bars) for t in ts]
    grid = (torch.cat(imgs) + 1) / 2                 # back to [0,1] for save_grid

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    save_grid(grid, os.path.join(ROOT, "results", "diffusion_forward.png"),
              nrow=6, titles=[f"t={t}" for t in ts])