"""
Train the low-light enhancement stage.

Zero-DCE trains WITHOUT paired ground truth (unsupervised, curve-estimation loss).
MIRNet trains WITH paired (low-light, normal-light) images (supervised, L1 + SSIM-ish loss).

Usage:
    # Zero-DCE (only needs low-light images, e.g. LOL dataset "low" folder or your own dark photos)
    python train_enhancement.py --model zero_dce --data_dir ./data/lowlight_only --epochs 100

    # MIRNet (needs paired low/high folders, e.g. LOL dataset: low/ and high/)
    python train_enhancement.py --model mirnet --low_dir ./data/LOL/low --high_dir ./data/LOL/high --epochs 100
"""

import os
import argparse
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

from zero_dce import ZeroDCE, zero_dce_loss
from mirnet import MIRNet


# ---------------------------- Datasets ----------------------------

class LowLightOnlyDataset(Dataset):
    """For Zero-DCE: only needs a folder of low-light images."""
    def __init__(self, root_dir, image_size=256):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class PairedLowHighDataset(Dataset):
    """For MIRNet: needs matching filenames in low_dir and high_dir."""
    def __init__(self, low_dir, high_dir, image_size=256):
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.*")))
        self.high_paths = sorted(glob.glob(os.path.join(high_dir, "*.*")))
        assert len(self.low_paths) == len(self.high_paths), \
            "low_dir and high_dir must have the same number of matching images"
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        high = Image.open(self.high_paths[idx]).convert("RGB")
        return self.transform(low), self.transform(high)


# ---------------------------- Training loops ----------------------------

def train_zero_dce(args, device):
    dataset = LowLightOnlyDataset(args.data_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = ZeroDCE(num_iterations=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        running = 0.0
        for imgs in loader:
            imgs = imgs.to(device)
            enhanced, curve_maps = model(imgs)
            loss, parts = zero_dce_loss(imgs, enhanced, curve_maps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running += loss.item()

        avg = running / max(len(loader), 1)
        print(f"[Zero-DCE] Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}")

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"zero_dce_epoch{epoch+1}.pt"))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "zero_dce_final.pt"))
    print(f"Saved final Zero-DCE weights to {args.out_dir}/zero_dce_final.pt")


def train_mirnet(args, device):
    dataset = PairedLowHighDataset(args.low_dir, args.high_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = MIRNet(channels=32, num_rrg=2, num_msrb=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    l1_loss = torch.nn.L1Loss()

    for epoch in range(args.epochs):
        running = 0.0
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            pred = model(low)
            loss = l1_loss(pred, high)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg = running / max(len(loader), 1)
        print(f"[MIRNet] Epoch {epoch+1}/{args.epochs}  L1 loss={avg:.4f}")

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"mirnet_epoch{epoch+1}.pt"))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "mirnet_final.pt"))
    print(f"Saved final MIRNet weights to {args.out_dir}/mirnet_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["zero_dce", "mirnet"], required=True)
    parser.add_argument("--data_dir", help="Low-light-only images folder (Zero-DCE)")
    parser.add_argument("--low_dir", help="Low-light images folder (MIRNet)")
    parser.add_argument("--high_dir", help="Normal-light ground truth folder (MIRNet)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--out_dir", default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.model == "zero_dce":
        assert args.data_dir, "--data_dir required for Zero-DCE"
        train_zero_dce(args, device)
    else:
        assert args.low_dir and args.high_dir, "--low_dir and --high_dir required for MIRNet"
        train_mirnet(args, device)
