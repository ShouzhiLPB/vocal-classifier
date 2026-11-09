import torch
import torch.nn as nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class VocalCNN(nn.Module):
    """Log-mel (B, n_mels, T) -> logits (B, n_classes).

    * Per-example normalisation: overall loudness is removed, so the model
      cannot cheat with "singing was recorded louder".
    * Global average pooling: any T >= 16 frames (0.16 s) is accepted.
    """

    def __init__(self, n_classes: int = 2, width=(16, 32, 64, 64)):
        super().__init__()
        chans = (1, *width)
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans, chans[1:])])
        self.head = nn.Linear(width[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        x = self.features(x.unsqueeze(1))
        return self.head(x.mean(dim=(2, 3)))
