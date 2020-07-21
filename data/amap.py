import torch
from torch.utils.data import Dataset
from PIL import Image

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

        all_frames = []
        for item in annotations["annotations"]:
            idx, key_frame, status = item["id"], item["key_frame"], item["status"]
            for frame in item["frames"]:
                frame_name = frame["frame_name"]
                if stage == "trainval" and frame_name != key_frame:
                    continue
                all_frames.append((idx, frame_name, frame_name == key_frame, status))

        if shuffle:
            random.shuffle(all_frames)

        self.all_frames = all_frames

        counter = collections.Counter([item[3] for item in all_frames])
        weights = {key: len(all_frames) / value for key, value in counter.items()}
        self.sampler_weights = [weights[item[3]] for item in all_frames]

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        idx, frame_name, is_key, status = self.all_frames[index]

        image_file = os.path.join(self.image_folder, str(idx), frame_name)
        image = Image.open(image_file)

        if self.transforms:
            image = self.transforms(image)

        if self.target_transforms:
            status = self.target_transforms(status)

        if self.is_test:
            return idx, image, is_key
        else:
            return image, status, is_key


if __name__ == '__main__':
    dataset = AmapDataset("/home/killf/dataset/amap_traffic/amap")
    for item in dataset:
        pass
