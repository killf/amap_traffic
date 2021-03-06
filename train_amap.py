import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet101, resnet50, resnet34
from torchvision.transforms import *
import numpy as np
import json
import os

from data.amap import AmapDataset
from utils import Counter
from config.amap import *


def create_model(num_classes=3, pretrained=True):
    model = resnet34(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    train_transforms = Compose([Resize((640, 320)), RandomHorizontalFlip(), RandomGrayscale(),
                                RandomCrop((640, 320), 20), ToTensor()])
    train_dataset = AmapDataset(DATA_DIR, "train", transforms=train_transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    device = torch.device(DEVICE)
    model = create_model(num_classes=3).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        counter = Counter()
        for step, (img, label, is_key) in enumerate(train_loader):
            step, total_step = step + 1, len(train_loader)
            img, label, is_key = img.to(device), label.to(device), is_key.to(device)

            pred = model(img)

            loss = nn.functional.cross_entropy(pred, label, reduction="none")
            loss_weight = (is_key.float() + 1) / 2
            loss_weight = loss_weight / torch.sum(loss_weight)
            loss = torch.dot(loss, loss_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(pred, 1)
            acc = (pred == label).float().mean().cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()

            counter.append(loss=loss, acc=acc)
            print(f"Epoch:{epoch}/{EPOCHS}, Step:{step}/{total_step}, "
                  f"Loss:{loss:.04f}/{counter.loss:.04f}, "
                  f"Accuracy:{acc:0.4f}/{counter.acc:.04f}",
                  end='\r', flush=True)

        torch.save(model.state_dict(), MODEL_FILE)
        lr_scheduler.step()
        print()

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
