from io import BytesIO
import os
from os import path as osp
from urllib import request
from zipfile import ZipFile


PROJ_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_DIR = osp.join(PROJ_ROOT, 'data')
STYLES_DIR = osp.join(DATA_DIR, 'style')
CONTENT_DIR = osp.join(DATA_DIR, 'content')

COCO_URL = 'http://images.cocodataset.org/zips/train2017.zip'


def _download_style_images():
    if not osp.isdir(STYLES_DIR):
        os.mkdir(STYLES_DIR)
    with open(osp.join(DATA_DIR, 'style_urls.txt')) as f_style_urls:
        for i, url in enumerate(f_style_urls):
            url = url.rstrip()
            out_file = osp.join(STYLES_DIR, f'{i}{osp.splitext(url)[1].lower()}')
            if osp.isfile(out_file):
                continue
            request.urlretrieve(url.rstrip(), out_file)
    request.urlcleanup()


def _download_content_images():
    with request.urlopen(COCO_URL) as f_coco:
        with ZipFile(BytesIO(f_coco.read()), 'r') as f_coco_zip:
            container_dir = f_coco_zip.infolist()[0].filename
            f_coco_zip.extractall(DATA_DIR)
    os.rename(osp.join(DATA_DIR, container_dir), CONTENT_DIR)


def main():
    if not osp.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    _download_style_images()
    _download_content_images()


if __name__ == '__main__':
    main()
