import io
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from config import Config

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


model = UNet().to(DEVICE)
if os.path.exists(Config.MODEL_PATH):
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
])


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": Config.MODEL_PATH,
        "device": str(DEVICE)
    })


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file with key 'image'"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    original_size = image.size

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float().squeeze().cpu().numpy()

    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=Config.PORT)