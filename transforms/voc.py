import torchvision.transforms.functional as F
import torch
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CvtLabel:
    def __init__(self, labels):
        self.labels = labels
        self.label2idx = {v: i for i, v in enumerate(labels)}

    def __call__(self, pic, target):
        labels, boxes = [], []

        ls = target['annotation']['object']
        if not isinstance(ls, list):
            ls = [ls]

        for item in ls:
            label = item['name']
            label = self.label2idx[label]
            labels.append(label)

            box = [float(item['bndbox'][n]) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(box)

        labels = np.array(labels, dtype=np.int64)
        boxes = np.array(boxes, dtype=np.float32)

        return pic, {'labels': labels, 'boxes': boxes}


class ToTensor:
    def __init__(self, device=None):
        if isinstance(device, int):
            device = torch.device(device)
        self.device = device

    def __call__(self, pic, target):
        pic = F.to_tensor(pic)

        target['labels'] = torch.from_numpy(target['labels'])
        target['boxes'] = torch.from_numpy(target['boxes'])

        if self.device:
            pic.to(self.device)
            target['labels'].to(self.device)
            target['boxes'].to(self.device)

        return pic, target

    def __repr__(self):
        return self.__class__.__name__ + '()'
