import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import faster_rcnn
import os

from transforms.voc import Compose, CvtLabel, ToTensor
from data.voc import VOCDetection
from config.voc import *

LABEL_NAMES = ['__background__', 'car', 'bus', 'van', 'others']


def collate_fn(batch):
    result = [[] for _ in range(len(batch[0]))]
    for data in batch:
        for i, item in enumerate(data):
            result[i].append(item)
    return result


def fasterrcnn_resnet50_fpn(num_classes=2, pretrained=True, pretrained_backbone=True):
    model = faster_rcnn.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    transforms = Compose([CvtLabel(LABEL_NAMES), ToTensor()])

    train_set = VOCDetection(DATA_DIR, "train_list.txt", transforms=transforms)
    val_set = VOCDetection(DATA_DIR, "trainval.txt", transforms=transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    device = torch.device(0)
    model = fasterrcnn_resnet50_fpn(num_classes=len(LABEL_NAMES), pretrained=True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if os.path.exists(MODEL_FILE):
        state_dict, _ = torch.load(MODEL_FILE)
        model.load_state_dict(state_dict)

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
            print(f"Epoch:{epoch}/{EPOCHS + 1}, Step:{step}/{total_step}, Loss={loss:.04f}")

        model.eval()
        for img, target in val_loader:
            img = [i.to(device) for i in img]
            target = [{n: item[n].to(device) for n in ['labels', 'boxes']} for item in target]

            pred = model(img)
            pass

        lr_scheduler.step()
        torch.save(model.state_dict(), MODEL_FILE)

    pass


if __name__ == '__main__':
    main()
