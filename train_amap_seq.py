import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os

from data.amap_seq import AmapDataset
from transforms.amap_seq import *
from models import resnet34, resnet50
from utils import Counter
from config.amap import *

MODEL_FILE = "amap_seq"
NUM_WORKERS = 0
BATCH_SIZE = 1


class MyModule(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MyModule, self).__init__()
        self.feature = resnet50(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, 0)

        x, (h_n, c_n) = self.lstm(x)

        x = h_n[-1, :, :]

        x = self.classifier(x)
        return x


def collate_fn(inputs):
    assert len(inputs) == 1

    size = len(inputs[0])
    result = [[] for _ in range(size)]
    for i, item in enumerate(inputs):
        for j in range(size):
            v = inputs[i][j]
            if isinstance(v, list):
                v = torch.stack(v)
            result[j].append(v)

    result[0] = np.array(result[0])
    result[1] = result[1][0]
    if size > 2:
        result[2] = torch.tensor(result[2])

    return result


def main():
    train_transforms = Compose([Resize((640, 320)), RandomHorizontalFlip(),
                                RandomCrop((640, 320), 20), ToTensor()])
    train_dataset = AmapDataset(DATA_DIR, "train", transforms=train_transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    device = torch.device(DEVICE)
    model = MyModule(num_classes=3).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        counter = Counter()
        for step, (idx, img, label) in enumerate(train_loader):
            step, total_step = step + 1, len(train_loader)
            img, label = img.to(device), label.to(device)

            pred = model(img)

            loss = nn.functional.cross_entropy(pred, label)

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


def test():
    transforms = Compose([Resize((640, 320)), ToTensor()])

    test_dataset = AmapDataset(DATA_DIR, "test", transforms=transforms)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device(DEVICE)
    model = MyModule(num_classes=3, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_FILE))

    model.eval()
    with torch.no_grad():
        pred_dict = {}
        for idx, img in test_loader:
            img = img.to(device)
            pred = model(img)
            pred = torch.argmax(pred, 1).cpu().numpy()

            for idx_, pred_ in zip(idx, pred):
                idx_ = str(idx_)
                pred_ = int(pred_)
                pred_dict[idx_] = pred_

    annotations = json.load(open(os.path.join(DATA_DIR, "amap_traffic_annotations_test.json")))
    for item in annotations["annotations"]:
        item['status'] = int(pred_dict[item["id"]])

    json.dump(annotations, open("result.json", "w", encoding="utf8"))


if __name__ == '__main__':
    # main()
    test()
