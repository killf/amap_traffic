import torch
import torchvision.transforms.functional as F
from PIL import Image
from collections import Iterable
import numbers
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        if isinstance(image, list):
            return [F.resize(img, self.size, self.interpolation) for img in image]
        else:
            return F.resize(image, self.size, self.interpolation)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            if isinstance(image, list):
                return [F.hflip(img) for img in image]
            return F.hflip(image)
        return image


class RandomCrop:
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image):
        if self.padding is not None:
            if isinstance(image, list):
                image = [F.pad(img, self.padding, self.fill, self.padding_mode) for img in image]
            else:
                image = F.pad(image, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            if isinstance(image, list):
                image = [F.pad(img, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode) for img in image]
            else:
                image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            if isinstance(image, list):
                image = [F.pad(img, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode) for img in image]
            else:
                image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        if isinstance(image, list):
            result = []
            for img in image:
                i, j, h, w = self.get_params(img, self.size)
                result.append(F.crop(img, i, j, h, w))
            return result

        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w)


class ToTensor:
    def __call__(self, pic):
        if isinstance(pic, list):
            return [F.to_tensor(img) for img in pic]
        return F.to_tensor(pic)


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))
