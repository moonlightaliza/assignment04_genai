import os, sys, json, time
import torch
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from starter.uhat import load_uhat_characters
from schedule import make_schedule, q_sample, T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = os.path.join(ROOT, "starter", "data", "uhat")
x_tr, y_tr, names = load_uhat_characters(root=data_root, split="train")
x_tr = x_tr * 2 - 1                                  # [0,1] -> [-1,1]
loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)

unet = UNet2DModel(...).to(device)                   
opt = torch.optim.Adam(unet.parameters(), lr=2e-4)
betas, alphas, alpha_bars = make_schedule(device)

for epoch in range(1, 41):
    total = 0.0
    for x0, y in loader:
        x0, y = x0.to(device), y.to(device)
        t = torch.randint((0, T, x0.size(0)), device = device, dtype = torch.long)
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, alpha_bars)
        eps_pred = unet(x_t, t, class_labels=y).sample
        loss = F.mse_loss(eps_pred, eps)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x0.size(0)
    print(epoch, total / len(x_tr))
    