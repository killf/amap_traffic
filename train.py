import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.models import resnet101
import numpy as np
import os

from data.amap import AmapDataset
from config.amap import *


def create_model(num_classes=3, pretrained=True):
    model = resnet101(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    transforms = Compose([Resize((640, 320)), ToTensor()])

    train_dataset = AmapDataset(DATA_DIR, "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    device = torch.device(DEVICE)
    model = create_model(num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    for epoch in range(1, EPOCHS + 1):
        model.train()

        all_loss, all_acc = [], []
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

            all_loss.append(loss)
            all_acc.append(acc)
            print(f"Epoch:{epoch}/{EPOCHS}, Step:{step}/{total_step}, "
                  f"Loss:{loss:.04f}/{np.mean(all_loss):.04f}, "
                  f"Accuracy:{acc:0.4f}/{np.mean(all_acc):.04f}",
                  end='\r', flush=True)

        lr_scheduler.step()
        torch.save(model.state_dict(), MODEL_FILE)
        print()


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

        idx_ls = np.stack(idx_ls)
        is_key_ls = np.stack(is_key_ls)
        pred_ls = np.stack(pred_ls)

        pass


if __name__ == '__main__':
    # main()
    test()
