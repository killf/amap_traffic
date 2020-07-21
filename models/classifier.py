import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes, backend=None):
        super(SimpleClassifier, self).__init__()

        self.backend = backend
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        if self.backend:
            x = self.backend(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
