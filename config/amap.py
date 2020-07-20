import os

DATA_DIR = "/home/killf/dataset/amap_traffic/amap"
MODEL_FILE = "result/amap.pth"

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 8
NUM_WORKERS = 7
DEVICE = 0
