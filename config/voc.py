import os

DATA_DIR = "/home/killf/dataset/amap_traffic/UA-DETRAC交通车辆数据集"
MODEL_FILE = "result/model.pth"

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 4
NUM_WORKERS = 7
