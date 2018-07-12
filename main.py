import argparse
from copy import deepcopy
from os import path as osp

import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms

from dataset import ImageFolder
from model import TransferNet, StyleTransfer


def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-images', type=osp.abspath, default='data/content')
    parser.add_argument('--style-images', type=osp.abspath, default='data/style')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner-epoch-size', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--outer-lr', type=float, default=0.1)
    parser.add_argument('--inner-lr', type=float, default=0.001)

    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--relu-type', choices=('relu', 'selu', 'leaky'),
                        default='leaky')
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--style-lambda', type=float, default=1)
    parser.add_argument('--content-lambda', type=float, default=1)
    parser.add_argument('--content-layer', type=int, choices=range(1, 5),
                        default=4)
    return parser.parse_args()


def main():
    opts = _get_opts()

    styler = TransferNet(num_channels=opts.channels, relu_type=opts.relu_type)
    style_transfer = StyleTransfer(styler,
                                   content_layer=opts.content_layer,
                                   style_lambda=opts.style_lambda,
                                   content_lambda=opts.content_lambda).cuda()

    styles_loader = _make_dataloader('style', opts)
    content_loader = _make_dataloader('content', opts)

    optim = torch.optim.Adam(style_transfer.styler.parameters(),
                             lr=opts.inner_lr,
                             betas=(0, 0))

    def _train_inner(style_images):
        epoch_loss = 0
        for i, content_images in enumerate(content_loader):
            if i * opts.batch_size > opts.inner_epoch_size:
                break
            loss = style_transfer(style_images, content_images).mean()
            epoch_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        return epoch_loss / i

    for epoch in range(1, opts.epochs, 1):
        for style in styles_loader:
            _imshow(style[0])
            # orig_weights = deepcopy(styler.state_dict())
            loss = _train_inner(style.cuda())
            # new_weights = styler.state_dict()
            # styler.load_state_dict({
            #     name: (orig_weights[name] +
            #            (new_weights[name] - orig_weights[name]) * opts.outer_lr)
            #     for name in orig_weights})

            with torch.no_grad():
                styled = styler(next(iter(content_loader)).cuda())
                _imshow(styled.cpu(), size=(8, 4))
            break
        print(f'[{epoch}] {loss:.3f}')


def _imshow(imgs, size=(1, 1)):  # size: (w, h)
    imgs = imgs * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if len(imgs.shape) == 4:
        grid = torchvision.utils.make_grid(imgs)
    else:
        grid = imgs
    grid = grid.permute(1, 2, 0)
    ax.imshow(grid, aspect='equal')
    plt.show()
    plt.close()


def _make_dataloader(style_or_content, opts):
    assert style_or_content in {'style', 'content'}
    txform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    imgs = ImageFolder(getattr(opts, f'{style_or_content}_images'), txform)
    is_content = style_or_content == 'content'
    batch_size = opts.batch_size if is_content else 1
    return torch.utils.data.DataLoader(imgs,
                                       batch_size=batch_size,
                                       shuffle=is_content,
                                       num_workers=opts.num_workers,
                                       pin_memory=True)


if __name__ == '__main__':
    main()
