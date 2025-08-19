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
#from aodnet_model import AODNet
from ffanet_model import FFA

# ---------------------------
# Custom Dataset
# ---------------------------
class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, transform=None, format='.png'):
        self.hazy_files = sorted(glob(os.path.join(hazy_dir, '*')))
        self.gt_dir = gt_dir
        self.transform = transform
        self.format = format

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy = Image.open(self.hazy_files[idx]).convert('RGB')
        img = self.hazy_files[idx]
        id = img.split('/')[-1].split('_')[0]
        gt_name = id + self.format
        gt = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

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

    # Model, loss, optimizer
    model = FFA(gps = 3, blocks = 19).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(args.device == "cuda"))

    # Training
    batch_size = args.batch_size
    for epoch in range(1, args.epochs + 1):
        success = False
        while not success and batch_size > 0:
            try:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
                model.train()
                running_loss = 0.0

                for i, (hazy, gt) in enumerate(train_loader):
                    hazy, gt = hazy.to(args.device), gt.to(args.device)

                    optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=(args.device == "cuda")):
                        output = model(hazy)
                        loss = criterion(output, gt)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_loss:.4f}")

                # Save checkpoint & sample
                if epoch % args.save_every == 0:
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"ffanet_epoch{epoch}.pth"))
                    save_image(output, os.path.join(args.checkpoint_dir, f"sample_epoch{epoch}.png"))

                success = True  # finished this epoch without OOM

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    print(f"[WARN] CUDA OOM → reducing batch size to {batch_size} and retrying...")
                else:
                    raise e

        if batch_size == 0:
            print("[ERROR] Batch size reduced to 0, cannot continue training.")
            break

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train(args)
