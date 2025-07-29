import os
import argparse
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from aodnet_model import AODNet

# ---------------------------
# Custom Dataset
# ---------------------------
class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, transform=None):
        self.hazy_files = sorted(glob(os.path.join(hazy_dir, '*')))
        self.gt_files = sorted(glob(os.path.join(gt_dir, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy = Image.open(self.hazy_files[idx]).convert('RGB')
        gt = Image.open(self.gt_files[idx]).convert('RGB')

        if self.transform:
            hazy = self.transform(hazy)
            gt = self.transform(gt)

        return hazy, gt


# ---------------------------
# Training Loop
# ---------------------------
def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data transform
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor()
    ])

    # Dataset and Loader
    train_dataset = DehazeDataset(args.train_hazy, args.train_gt, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model, loss, optimizer
    model = AODNet().to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (hazy, gt) in enumerate(train_loader):
            hazy, gt = hazy.to(args.device), gt.to(args.device)

            output = model(hazy)
            loss = criterion(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_loss:.4f}")

        # Save checkpoint & sample
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"aodnet_epoch{epoch}.pth"))
            save_image(output, os.path.join(args.checkpoint_dir, f"sample_epoch{epoch}.png"))

    print("Training complete.")


# ---------------------------
# Argument Parser
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_hazy", type=str, required=True, help="Path to hazy images for training")
    parser.add_argument("--train_gt", type=str, required=True, help="Path to clean GT images for training")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--img_size", type=int, default=1024, help="Resize images to this size")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train(args)
