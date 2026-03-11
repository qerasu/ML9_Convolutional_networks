import torch
import timm
import torch.nn as nn

_LENET_INPUT_SIZE = 64

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),         # 64 → 60
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 60 → 30
            nn.Conv2d(6, 16, kernel_size=5),        # 30 → 26
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)   # 26 → 13
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes)
        )

    def forward(self, x: torch.Tensor):
        if x.shape[2] != _LENET_INPUT_SIZE or x.shape[3] != _LENET_INPUT_SIZE:
            raise ValueError(
                f"LeNet5 expects {_LENET_INPUT_SIZE}×{_LENET_INPUT_SIZE} input, "
                f"got {x.shape[2]}×{x.shape[3]}"
            )
        x = self.features(x)
        x = self.classifier(x)

        return x


def create_pretrained_model(
    backbone_name: str = "resnet18",
    num_classes: int = 9,
    pretrained: bool = True
):
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes
    )