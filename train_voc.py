import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
import os

from transforms.voc import Compose, CvtLabel, ToTensor
from config.voc import *

LABEL_NAMES = ['__background__', 'car', 'bus', 'van', 'others']


def collate_fn(batch):
    result = [[] for _ in range(len(batch[0]))]
    for data in batch:
        for i, item in enumerate(data):
            result[i].append(item)
    return result


def main():
    transforms = Compose([CvtLabel(LABEL_NAMES), ToTensor()])

    train_set = VOCDetection(DATA_DIR, 'UA-DETRAC', "train", transforms=transforms)
    val_set = VOCDetection(DATA_DIR, 'UA-DETRAC', "trainval", transforms=transforms)

    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    device = torch.device(0)
    model = fasterrcnn_resnet50_fpn(num_classes=len(LABEL_NAMES)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, (img, target) in enumerate(train_loader):
            step, total_step = step + 1, len(train_loader)

            img = [i.to(device) for i in img]
            target = [{n: item[n].to(device) for n in ['labels', 'boxes']} for item in target]

            loss_dict = model(img, target)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss = losses.cpu().detach().numpy()
            print(f"Epoch:{epoch}/{EPOCHS}, Step:{step}/{total_step}, Loss={loss:.04f}", end='\r', flush=True)

        # model.eval()
        # with torch.no_grad():
        #     for img, target in val_loader:
        #         img = [i.to(device) for i in img]
        #         target = [{n: item[n].to(device) for n in ['labels', 'boxes']} for item in target]
        #
        #         pred = model(img)
        #         for item in pred:
        #             print(item["boxes"].shape, item['labels'].shape, item['scores'].shape)
        #             pass
        #
        #         pass

        lr_scheduler.step()
        torch.save(model.state_dict(), MODEL_FILE)
        print()

    pass


if __name__ == '__main__':
    main()
