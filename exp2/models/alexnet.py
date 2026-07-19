"""The original AlexNet architecture described in lecture 3.

The model expects 227 x 227 RGB images and produces logits for the 1,000
ImageNet classes.  Conv2, Conv4 and Conv5 retain the two-way grouping used by
the original dual-GPU implementation.
"""

import torch
import torch.nn as nn


NUM_CLASSES = 1000
INPUT_SIZE = 227


class AlexNet(nn.Module):
    """Original 2012 AlexNet for ImageNet classification."""

    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # [N, 3, 227, 227] -> [N, 96, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            # [N, 96, 55, 55] -> [N, 96, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Each of the two original GPU groups receives 48 channels and
            # produces 128 channels: 96 -> 256 in total.
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            # [N, 256, 27, 27] -> [N, 256, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # [N, 256, 13, 13] -> [N, 256, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def test():
    """Run a small structural smoke test without loading a dataset."""
    model = AlexNet()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
    print("Input shape: ", (1, 3, INPUT_SIZE, INPUT_SIZE))
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    test()
