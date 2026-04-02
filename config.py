import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PORT = int(os.getenv("PORT", "5002"))
    MODEL_PATH = os.getenv("MODEL_PATH", "outputs/checkpoints/best_model.pth")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    EPOCHS = int(os.getenv("EPOCHS", "10"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    TRAIN_IMAGES = os.getenv("TRAIN_IMAGES", "data/processed/train/images")
    TRAIN_MASKS = os.getenv("TRAIN_MASKS", "data/processed/train/masks")
    VAL_IMAGES = os.getenv("VAL_IMAGES", "data/processed/val/images")
    VAL_MASKS = os.getenv("VAL_MASKS", "data/processed/val/masks")
    TEST_IMAGES = os.getenv("TEST_IMAGES", "data/processed/test/images")
    TEST_MASKS = os.getenv("TEST_MASKS", "data/processed/test/masks")