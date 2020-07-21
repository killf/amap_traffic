from torchvision.datasets import voc
import os

voc.DATASET_YEAR_DICT['UA-DETRAC'] = {
    'url': '',
    'filename': '',
    'md5': '',
    'base_dir': 'VOC2007'
}

DATA_DIR = "/home/killf/dataset/amap_traffic/UA-DETRAC交通车辆数据集"
MODEL_FILE = "result/detector.pth"

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 2
NUM_WORKERS = 2
DEVICE = 0