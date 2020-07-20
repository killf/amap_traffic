import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.models import resnet101
import numpy as np
import os

from data.amap import AmapDataset
from config.amap import *


def create_model(num_classes=1, pretrained=True):
    model = resnet101(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    transforms = Compose([Resize((640, 320)), ToTensor()])

    train_dataset = AmapDataset(DATA_DIR, "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    device = torch.device(DEVICE)
    model = create_model(num_classes=1).to(device)
    loss_fn = nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    for epoch in range(1, EPOCHS + 1):
        model.train()

        all_loss = []
        for step, (img, score, is_key) in enumerate(train_loader):
            step, total_step = step + 1, len(train_loader)
            img, score = img.to(device), score.to(device).type(torch.float32)

            pred = model(img)
            pred = torch.sigmoid(pred).squeeze()

            losses = loss_fn(pred, score)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss = losses.cpu().detach().numpy()
            all_loss.append(loss)
            mean_loss = np.mean(all_loss)
            print(f"Epoch:{epoch}/{EPOCHS}, Step:{step}/{total_step}, Loss:{loss:.04f}, Mean Loss:{mean_loss:.04f}",
                  end='\r', flush=True)

        lr_scheduler.step()
        torch.save(model.state_dict(), MODEL_FILE)
        print()


if __name__ == '__main__':
    main()
