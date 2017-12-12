import os

from chainer.dataset import dataset_mixin
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize
from chainercv.utils import read_image


class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, resize_to, crop_to, flip=True):
        self.path = path
        self.flip = flip
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.ids = [os.path.splitext(file)[0] for file in
                    os.listdir(self.path)]

    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return os.path.join(self.path, '{:s}.jpg'.format(self.ids[i]))

    def get_example(self, i):
        img = read_image(self.get_img_path(i))
        img = img.astype('f')
        img = img * 2 / 255.0 - 1.0  # [-1, 1)

        img = resize(img, (self.resize_to, self.resize_to))
        if self.resize_to > self.crop_to:
            img = random_crop(img, (self.crop_to, self.crop_to))
        if self.flip:
            img = random_flip(img, x_random=True)
        return img
