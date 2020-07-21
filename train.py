import json

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet101
from torchvision.transforms import *
import numpy as np
import os

from data.amap import AmapDataset
from utils import Counter
from config.amap import *


def create_model(num_classes=3, pretrained=True):
    model = resnet101(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    train_transforms = Compose(
        [Resize((640, 320)), RandomHorizontalFlip(), RandomGrayscale(), RandomCrop((640, 320), 20), ToTensor()])
    val_transforms = Compose([Resize((640, 320)), ToTensor()])

    train_dataset = AmapDataset(DATA_DIR, "train", transforms=train_transforms)
    val_dataset = AmapDataset(DATA_DIR, "trainval", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    device = torch.device(DEVICE)
    model = create_model(num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        counter = Counter()
        for step, (img, label, is_key) in enumerate(train_loader):
            step, total_step = step + 1, len(train_loader)
            img, label = img.to(device), label.to(device)

            pred = model(img)
            losses = loss_fn(pred, label)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            pred = torch.argmax(pred, 1)
            acc = (pred == label).float().mean().cpu().detach().numpy()
            loss = losses.cpu().detach().numpy()

            counter.append(loss=loss, acc=acc)
            print(f"Epoch:{epoch}/{EPOCHS}, Step:{step}/{total_step}, "
                  f"Loss:{loss:.04f}/{counter.loss:.04f}, "
                  f"Accuracy:{acc:0.4f}/{counter.acc:.04f}",
                  end='\r', flush=True)

        model.eval()
        counter = Counter()
        with torch.no_grad():
            for img, label, is_key in val_loader:
                img, label = img.to(device), label.to(device)

                pred = model(img)
                losses = loss_fn(pred, label)

                pred = torch.argmax(pred, 1)
                acc = (pred == label).float().mean().cpu().detach().numpy()
                loss = losses.cpu().detach().numpy()

                counter.append(loss=loss, acc=acc)

        val_acc, val_loss = counter.acc, counter.loss
        print(f"\nVal Loss:{val_loss:.04f} Acc:{val_acc:.04f}\n")

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_FILE)

        lr_scheduler.step()
    print(f"Best Acc:{best_acc:.04f}")


def test():
    transforms = Compose([Resize((640, 320)), ToTensor()])

    test_dataset = AmapDataset(DATA_DIR, "test", transforms=transforms)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)

    device = torch.device(DEVICE)
    model = create_model(num_classes=3, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_FILE))

    model.eval()
    with torch.no_grad():
        idx_ls, is_key_ls, pred_ls = [], [], []
        for idx, img, is_key in test_loader:
            img = img.to(device)
            pred = model(img)
            pred = torch.argmax(pred, 1).cpu().numpy()
            idx_ls.append(idx)
            is_key_ls.append(is_key.numpy())
            pred_ls.append(pred)

        idx_ls = np.concatenate(idx_ls, axis=0)
        is_key_ls = np.concatenate(is_key_ls, axis=0)
        pred_ls = np.concatenate(pred_ls, axis=0)

    pred_dict = dict()
    for idx, is_key, pred in zip(idx_ls, is_key_ls, pred_ls):
        if not is_key:
            continue
        pred_dict[idx] = pred

    annotations = json.load(open(os.path.join(DATA_DIR, "amap_traffic_annotations_test.json")))
    for item in annotations["annotations"]:
        item['status'] = int(pred_dict[item["id"]])

    json.dump(annotations, open("result.json", "w", encoding="utf8"))


if __name__ == '__main__':
    # main()
    test()
