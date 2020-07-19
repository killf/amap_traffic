from torchvision.datasets import VisionDataset
from PIL import Image

import os
import sys
import collections

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

__ALL__ = ["VOCDetection"]


class VOCDetection(VisionDataset):
    def __init__(self,
                 root,
                 file,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.images, self.annotations = [], []
        with open(os.path.join(root, file), "r") as f:
            for line in f.readlines():
                ss = line.strip().split(" ")
                self.images.append(os.path.join(root, ss[0]))
                self.annotations.append(os.path.join(root, ss[1]))

        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
