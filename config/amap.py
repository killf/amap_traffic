import os

DATA_DIR = "/home/killf/dataset/amap_traffic/amap"
MODEL_FILE = "result/amap.pth"

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 2
NUM_WORKERS = 2
DEVICE = 0
