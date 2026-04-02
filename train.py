import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config
from metrics import dice_score, iou_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))

        x = self.up3(b)
        x = torch.cat([x, d3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.conv1(x)

        return self.out(x)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    train_ds = SegmentationDataset(Config.TRAIN_IMAGES, Config.TRAIN_MASKS, Config.IMAGE_SIZE)
    val_ds = SegmentationDataset(Config.VAL_IMAGES, Config.VAL_MASKS, Config.IMAGE_SIZE)
    test_ds = SegmentationDataset(Config.TEST_IMAGES, Config.TEST_MASKS, Config.IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []

    best_dice = 0.0

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_PATH)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig("outputs/plots/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(val_dices, label="Val Dice")
    plt.plot(val_ious, label="Val IoU")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.savefig("outputs/plots/metrics_curve.png")
    plt.close()

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=DEVICE))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()