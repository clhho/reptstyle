import argparse
from copy import deepcopy
from os import path as osp

import torch
from torch import nn
import torchvision
from torchvision import transforms

from dataset import ImageFolder
from model import TransferNet


def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-images', type=osp.abspath, default='data/content')
    parser.add_argument('--style-images', type=osp.abspath, default='data/styles')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--inner-epoch-size', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--outer-lr', type=float, default=0.01)
    parser.add_argument('--inner-lr', type=float, default=0.001)

    parser.add_argument('--channels', type=int, default=24)
    return parser.parse_args()


def main():
    opts = _get_opts()

    model = TransferNet(num_channels=opts.channels).cuda()

    get_styles, get_content = _make_feature_getters(opts)

    styles_loader = _make_dataloader('style', opts)
    content_loader = _make_dataloader('content', opts)

    optim = torch.optim.Adam(model.parameters(), lr=opts.inner_lr, betas=(0, 0))

    for _epoch in range(1, opts.epochs, 1):
        for style in styles_loader:
            orig_weights = deepcopy(model.state_dict())
            for i, imgs in enumerate(content_loader, 1):
                if i * opts.batch_size > opts.inner_epoch_size:
                    break
                styled = model(imgs.cuda())

                target_style_features = get_styles(style)
                style_features = get_styles(styled)
                target_gram = _gram_matrix(target_style_features)
                styles_gram = _gram_matrix(style_features)
                style_loss = ((styles_gram - target_gram)**2).mean(0).sum()

                target_content_features = get_content(imgs)
                content_features = get_content(styled)
                content_loss = ((content_features - target_content_features)**2).mean()

                loss = style_loss + content_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

            new_weights = model.state_dict()
            model.load_state_dict({
                name: (orig_weights[name]
                       (new_weights[name] - orig_weights[name]) * opts.outer_lr)
                for name in orig_weights})


def _make_dataloader(style_or_content, opts):
    assert style_or_content in {'style', 'content'}
    txform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    imgs = ImageFolder(getattr(opts, f'{style_or_content}_images'), txform)
    is_content = style_or_content == 'content'
    batch_size = opts.batch_size if is_content else 1
    return torch.utils.data.DataLoader(imgs,
                                       batch_size=batch_size,
                                       shuffle=is_content,
                                       num_workers=opts.num_workers,
                                       pin_memory=True)


def _make_feature_getters(_opts):
    vgg = torchvision.models.vgg16(pretrained=True).features.cuda()
    maxpool_inds = [i for i, l in enumerate(vgg) if isinstance(l, nn.MaxPool2d)]
    relus = [vgg[:i] for i in maxpool_inds[:-1]] # 1_2, 2_2, 3_3, 4_3
    def _get_styles(imgs):
        with torch.no_grad():
            return [relu(imgs) for relu in relus]
    def _get_contents(imgs):
        with torch.no_grad():
            return relus[2](imgs)
    return _get_styles, _get_contents


def _gram_matrix(features):
    features = features.reshape(*features.shape[:2], -1)
    num_features = features.size(1) * features.size(2)
    return features.bmm(features.transpose(1, 2)) / num_features


if __name__ == '__main__':
    # main()
    _make_feature_getters(_get_opts())
