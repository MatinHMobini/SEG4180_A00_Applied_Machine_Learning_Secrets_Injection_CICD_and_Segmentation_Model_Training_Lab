import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from train import UNet
from config import Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict_mask(model, image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float().squeeze().cpu().numpy()

    return image, pred

def main():
    os.makedirs("outputs/predictions", exist_ok=True)
    test_images = sorted(Path(Config.TEST_IMAGES).glob("*"))[:5]
    model = load_model()

    for img_path in test_images:
        image, pred = predict_mask(model, img_path)

        plt.figure()
        plt.imshow(image)
        plt.imshow(pred, alpha=0.4)
        plt.axis("off")
        plt.title(f"Prediction: {img_path.name}")
        plt.savefig(f"outputs/predictions/{img_path.stem}_overlay.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()