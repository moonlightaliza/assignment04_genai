import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

sys.path.append("../starter")
from starter.eurosat import load_eurosat_split, patchify
from model import MaskedAutoEncoder


def train_mae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 256
    num_epochs = 100
    lr = 5e-4
    weight_decay = 0.05

    print("Loading EuroSAT training split...")
    train_images, _ = load_eurosat_split("train")      # labels ignored
    train_patches = patchify(train_images)             # (16200, 64, 192)

    dataset = TensorDataset(train_patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = MaskedAutoEncoder().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses = []
    model.train()
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            loss, _, _ = model(batch)          # masked-only MSE computed inside forward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        epoch_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}] | Masked MSE: {avg_loss:.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "mae_pretrained.pth")   # safety checkpoint

    train_time = time.time() - start
    torch.save(model.state_dict(), "mae_pretrained.pth")
    print(f"Done in {train_time/60:.1f} min")

    os.makedirs("../results", exist_ok=True)

    # per-epoch log + Table 2 info
    with open("../results/mae_pretrain_log.json", "w") as f:
        json.dump({"epoch_losses": epoch_losses,
                   "params": num_params,
                   "train_time_sec": train_time,
                   "final_loss": epoch_losses[-1]}, f)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, color="blue", label="Pretraining loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Masked reconstruction MSE")
    plt.title("MAE pretraining loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("../results/mae_pretrain_loss.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_mae()