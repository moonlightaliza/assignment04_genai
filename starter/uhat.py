
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_split_dir(root, kind, split):
    for d in sorted(Path(root).rglob("*")):
        name = d.name.lower()

        if d.is_dir() and kind in name and split in name:
            if any(c.is_dir() for c in d.iterdir()):
                return d

    found = [
        str(d)
        for d in Path(root).rglob("*")
        if d.is_dir()
    ][:20]

    raise FileNotFoundError(
        f"No '{kind}'/'{split}' folder under {root}. "
        f"Folders seen: {found}"
    )


def load_uhat_characters(
    root="starter/data/uhat",
    split="train",
    class_names=None
):
    split_dir = _find_split_dir(root, "char", split)

    if class_names is None:
        class_names = sorted(
            c.name.lower()
            for c in split_dir.iterdir()
            if c.is_dir()
        )

    index = {
        name: i
        for i, name in enumerate(class_names)
    }

    images, labels = [], []

    for cdir in sorted(split_dir.iterdir()):
        if not cdir.is_dir() or cdir.name.lower() not in index:
            continue

        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                im = (
                    Image.open(f)
                    .convert("L")
                    .resize((28, 28))
                )

                images.append(
                    np.asarray(im, dtype=np.float32) / 255.0
                )

                labels.append(index[cdir.name.lower()])

    if not images:
        raise ValueError(f"No character images found in {split_dir}")

    x = torch.from_numpy(
        np.stack(images)
    ).unsqueeze(1)

    flip = x.mean(dim=(1, 2, 3)) > 0.5
    x[flip] = 1.0 - x[flip]

    return (
        x,
        torch.tensor(labels, dtype=torch.long),
        class_names
    )
