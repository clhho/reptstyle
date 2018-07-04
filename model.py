import torch
from torch import nn
import torch.nn.functional as nnf

class TransferNet(nn.Module):
    def __init__(self, num_channels=24):
        super().__init__()

        self.num_channels = num_channels

        layers = []
        for i in range(7):
            layers.extend(self._layer(self.num_channels if i > 0 else 3, 3, 2**i))
            layers.extend(self._layer(self.num_channels, 3, 1))
            layers.append(nn.Conv2d(self.num_channels, 3, 1))

        self.model = nn.Sequential(*layers)

    def _layer(self, in_channels, kernel_size, dilation):
        return [
            nn.Conv2d(in_channels, self.num_channels, kernel_size,
                      padding=(kernel_size - 1) // 2 + dilation - 1,
                      dilation=dilation),
            nn.InstanceNorm2d(self.num_channels),
            nn.LeakyReLU(),
        ]

    def forward(self, input):
        return self.model(input)


def test_transfer_net():
    net = TransferNet()
    inp = torch.randn(2, 3, 224, 224)
    print(net(inp).shape)
