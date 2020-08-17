import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import collections
import random
import json
import os


class AmapDataset(Dataset):
    def __init__(self, root, stage="train", shuffle=False, transforms=None, target_transforms=None):
        self.root = root
        self.image_folder = os.path.join(root, f"amap_traffic_{stage}_0712")
        self.annotations_file = os.path.join(root, f"amap_traffic_annotations_{stage}.json")
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.is_test = stage == "test"

        annotations = json.load(open(self.annotations_file, encoding='utf8'))
        self.annotations = annotations["annotations"]

        if shuffle:
            random.shuffle(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]

        idx, status, images, times = annotation["id"], annotation["status"], [], []
        for frame in annotation["frames"]:
            frame_name, gps_time = frame["frame_name"], frame["gps_time"]
            image_file = os.path.join(self.image_folder, str(idx), frame_name)
            image = Image.open(image_file)
            images.append(image)
            times.append(torch.tensor(gps_time, dtype=torch.int32))

        if self.transforms:
            images = self.transforms(images)

        if self.target_transforms:
            status = self.target_transforms(status)

        if self.is_test:
            return idx, images, times
        else:
            return idx, images, times, status


if __name__ == '__main__':
    dataset = AmapDataset("/home/killf/dataset/amap_traffic/amap")
    for item in dataset:
        pass
