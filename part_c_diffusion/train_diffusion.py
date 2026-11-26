import os, sys, json, time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from starter.uhat import load_uhat_characters
from schedule import make_schedule, q_sample, T

# Drive on Colab, local folder otherwise
if os.path.exists("/content/drive/MyDrive"):
    SAVE_DIR = "/content/drive/MyDrive/genai_a4/part_c"
else:
    SAVE_DIR = os.path.join(ROOT, "part_c_diffusion")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 40          # set to 1 for the first test run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

data_root = os.path.join(ROOT, "starter", "data", "uhat")
x_tr, y_tr, names = load_uhat_characters(root=data_root, split="train")
x_tr = x_tr * 2 - 1                                   # [0,1] -> [-1,1]
loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)

unet = UNet2DModel(
    sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    num_class_embeds=40).to(device)
num_params = sum(p.numel() for p in unet.parameters())
print("parameters:", f"{num_params:,}")

opt = torch.optim.Adam(unet.parameters(), lr=2e-4)
betas, alphas, alpha_bars = make_schedule(device)

# resume if a checkpoint exists
ckpt_path = os.path.join(SAVE_DIR, "ckpt.pth")
start_epoch, losses, elapsed = 1, [], 0.0
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    losses, elapsed, start_epoch = ck["losses"], ck["elapsed"], ck["epoch"] + 1
    print("resuming from epoch", start_epoch)

unet.train()
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    t0 = time.time()
    total = 0.0
    for x0, y in loader:
        x0, y = x0.to(device), y.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)   # one random t per image
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, alpha_bars)
        eps_pred = unet(x_t, t, class_labels=y).sample
        loss = F.mse_loss(eps_pred, eps)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x0.size(0)

    losses.append(total / len(x_tr))
    elapsed += time.time() - t0
    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | loss {losses[-1]:.4f}")

    # checkpoint every epoch (survives Colab disconnects)
    torch.save({"epoch": epoch, "model": unet.state_dict(), "opt": opt.state_dict(),
                "losses": losses, "elapsed": elapsed}, ckpt_path)
    torch.save(unet.state_dict(), os.path.join(SAVE_DIR, "diffusion.pth"))
    with open(os.path.join(SAVE_DIR, "diffusion_log.json"), "w") as f:
        json.dump({"epoch_losses": losses, "params": num_params,
                   "train_time_sec": elapsed, "final_loss": losses[-1]}, f)

print("done")