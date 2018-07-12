import torch
from torch import nn
import torch.nn.functional as nnf
import torchvision


class StyleTransfer(nn.Module):
    def __init__(self, styler, content_layer, style_lambda, content_lambda):
        super().__init__()

        self.styler = styler
        self.style_lambda = style_lambda
        self.content_lambda = content_lambda
        self.content_layer = content_layer

        vgg = torchvision.models.vgg16(pretrained=True).features
        vgg.eval()

        # relu_1_2, 2_2, 3_3, 4_3, 5_3
        mp_inds = [i for i, l in enumerate(vgg) if isinstance(l, nn.MaxPool2d)]
        self._fex = nn.ModuleList()
        for i in range(len(mp_inds) - 1):
            ind = mp_inds[i]
            prev_ind = mp_inds[i - 1] if i > 0 else 0
            self._fex.append(vgg[prev_ind:ind])

    def forward(self, style_images, content_images):
        content_images = content_images.cuda()
        styled = self.styler(content_images)

        style_features = self._extract_features(styled)
        content_features = style_features[self.content_layer - 1]
        styles_gram = [self._gram_matrix(f) for f in style_features]

        with torch.no_grad():
            target_style_features = self._get_style_features(style_images)
            target_gram = [self._gram_matrix(f) for f in target_style_features]
            target_content_features = self._get_content_features(content_images)

        content_loss = ((content_features - target_content_features)**2).mean()
        style_loss = sum(((sg - tg.detach())**2).mean(0).sum()
                         for sg, tg in zip(styles_gram, target_gram))

        return self.style_lambda * style_loss + self.content_lambda * content_loss

    def _get_style_features(self, images):
        return self._extract_features(images)

    def _get_content_features(self, images):
        return self._extract_features(images)[self.content_layer - 1]

    def _extract_features(self, images):
        features = []
        inp = images
        for fex in self._fex:
            inp = fex(inp)
            features.append(inp)
        return features

    def _gram_matrix(self, features):
        batch_size, channels, w, h = features.shape
        features = features.reshape(batch_size, channels, -1)
        return features.bmm(features.transpose(1, 2)) / (channels * w * h)


class Layer(nn.Module):
    _RELU_TYPES = {
        'relu': nn.ReLU,
        'selu': nn.SELU,
        'leaky': lambda: nn.LeakyReLU(.1),
    }
    def __init__(self, num_channels, dilation,
                 relu_type='leaky', kernel_size=7, affine=False):
        super().__init__()

        kpad = (kernel_size - 1) // 2
        relu = self._RELU_TYPES[relu_type]

        self.resid = nn.Sequential(
            nn.InstanceNorm2d(num_channels, affine=affine),
            nn.Conv2d(num_channels, num_channels, kernel_size,
                      padding=kpad + kpad * (dilation - 1),
                      dilation=dilation, bias=affine),
            nn.InstanceNorm2d(num_channels, affine=affine),
            nn.Conv2d(num_channels, num_channels, kernel_size,
                      padding=kpad, bias=affine),
            relu(),
        )

    def forward(self, input):
        return input + self.resid(input)


class TransferNet(nn.Module):
    def __init__(self, num_channels, relu_type='relu', affine=False):
        super().__init__()

        self.num_channels = num_channels
        self.register_buffer('_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

        layers = [nn.Conv2d(3, self.num_channels, 3, padding=1)]
        for i in range(7):
            layers.append(Layer(num_channels, 2**i,
                                relu_type=relu_type, affine=affine))
        layers.append(Layer(num_channels, 1,
                            relu_type=relu_type, affine=affine))
        layers.extend([
            nn.InstanceNorm2d(self.num_channels, affine=affine),
            nn.Conv2d(self.num_channels, 3, 1, bias=affine),
            nn.Sigmoid(),
        ])

        for layer in layers:
            if not isinstance(layer, nn.Conv2d):
                continue
            w = layer.weight
            mid = (w.shape[-1] - 1) // 2
            nn.init.eye_(w[:, :, mid, mid])

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return (out - self._mean) / self._std


def test_transfer_net():
    net = TransferNet(num_channels=4)
    print(net)
    inp = torch.randn(2, 3, 256, 256)
    print(net(inp).shape)
