import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection.faster_rcnn import *
from torchvision.transforms import *
import numpy as np
import json
import os

from models import resnet50, SimpleClassifier, resnet_fpn_backbone
from data.amap import AmapDataset
from transforms.voc import CvtLabel, ToTensor as voc_ToTensor, Compose as voc_Compose
from utils import Counter
from config import voc, amap

LABEL_NAMES = ['__background__', 'car', 'bus', 'van', 'others']


def create_model(num_classes=3, pretrained=True):
    backbone = resnet50(pretrained=True)
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    backbone_fpn = resnet_fpn_backbone(backbone)
    detector = FasterRCNN(backbone_fpn, len(LABEL_NAMES))

    classifier = SimpleClassifier(num_classes, backbone)

    return detector, classifier


def create_data():
    def collate_fn(batch):
        result = [[] for _ in range(len(batch[0]))]
        for data in batch:
            for i, item in enumerate(data):
                result[i].append(item)
        return result

    voc_transforms = voc_Compose([CvtLabel(LABEL_NAMES), voc_ToTensor()])
    voc_train_set = VOCDetection(voc.DATA_DIR, 'UA-DETRAC', "train", transforms=voc_transforms)
    voc_train_loader = DataLoader(voc_train_set, voc.BATCH_SIZE, True, num_workers=voc.NUM_WORKERS,
                                  collate_fn=collate_fn)

    amap_train_transforms = Compose(
        [Resize((640, 320)), RandomHorizontalFlip(), RandomGrayscale(), RandomCrop((640, 320), 20), ToTensor()])
    amap_val_transforms = Compose([Resize((640, 320)), ToTensor()])

    amap_train_dataset = AmapDataset(amap.DATA_DIR, "train", transforms=amap_train_transforms)
    amap_val_dataset = AmapDataset(amap.DATA_DIR, "trainval", transforms=amap_val_transforms)

    amap_train_loader = DataLoader(amap_train_dataset, amap.BATCH_SIZE, True, num_workers=amap.NUM_WORKERS)
    amap_val_loader = DataLoader(amap_val_dataset, amap.BATCH_SIZE, num_workers=amap.NUM_WORKERS)

    return voc_train_loader, amap_train_loader, amap_val_loader


def main():
    voc_train_loader, amap_train_loader, amap_val_loader = create_data()

    device = torch.device(amap.DEVICE)
    detector, classifier = create_model()
    detector.to(device), classifier.to(device)

    params = [p for p in detector.parameters() if p.requires_grad]
    params += [p for p in classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in range(1, amap.EPOCHS + 1):
        detector.train()
        classifier.train()
        counter, step = Counter(), 0
        for step, (voc_data, amap_data) in enumerate(zip(voc_train_loader, amap_train_loader)):
            step, total_step = step + 1, min(len(voc_train_loader), len(amap_train_loader))

            img, target = voc_data
            img = [i.to(device) for i in img]
            target = [{n: item[n].to(device) for n in ['labels', 'boxes']} for item in target]

            voc_loss_dict = detector(img, target)
            voc_loss = sum(loss for loss in voc_loss_dict.values())

            img, label, is_key = amap_data
            img, label, is_key = img.to(device), label.to(device), is_key.to(device)
            amap_pred = classifier(img)
            amap_loss = F.cross_entropy(amap_pred, label, reduction="none")

            loss_weight = (is_key.float() + 1) / 2
            loss_weight = loss_weight / torch.sum(loss_weight)
            amap_loss = torch.dot(amap_loss, loss_weight)

            optimizer.zero_grad()
            voc_loss.backward()
            amap_loss.backward()
            optimizer.step()

            voc_loss = voc_loss.cpu().detach().numpy()

            amap_pred = torch.argmax(amap_pred, 1)
            amap_acc = (amap_pred == label).float().mean().cpu().detach().numpy()
            amap_loss = amap_loss.cpu().detach().numpy()

            counter.append(voc_loss=voc_loss, amap_loss=amap_loss, amap_acc=amap_acc)
            print(f"Epoch:{epoch}/{amap.EPOCHS}, Step:{step}/{total_step}, "
                  f"Detector Loss:{voc_loss:.04f}/{counter.voc_loss:.04f}, "
                  f"Classifier Loss:{amap_loss:.04f}/{counter.amap_loss:.04f}, "
                  f"Classifier Accuracy:{amap_acc:0.4f}/{counter.amap_acc:.04f}",
                  end='\r', flush=True)

        classifier.eval()
        counter = Counter()
        with torch.no_grad():
            for img, label, is_key in amap_train_loader:
                img, label = img.to(device), label.to(device)

                pred = classifier(img)
                losses = F.cross_entropy(pred, label)

                pred = torch.argmax(pred, 1)
                acc = (pred == label).float().mean().cpu().detach().numpy()
                loss = losses.cpu().detach().numpy()

                counter.append(loss=loss, acc=acc)

        val_acc, val_loss = counter.acc, counter.loss
        print(f"\nVal Loss:{val_loss:.04f} Acc:{val_acc:.04f}\n")

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), amap.MODEL_FILE)
            torch.save(detector.state_dict(), voc.MODEL_FILE)

        lr_scheduler.step()


if __name__ == '__main__':
    main()
