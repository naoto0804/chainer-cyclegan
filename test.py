#!/usr/bin/env python

import argparse
import os

import chainer.cuda
from chainer import serializers
from chainercv.transforms import resize
from chainercv.utils import read_image
from chainercv.utils import write_image

import net
from dataset import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', '-c', default='Generator',
                        help='Default generator class')
    parser.add_argument("--load_gen_model", default='',
                        help='load generator model')
    parser.add_argument('--out', '-o', default='output',
                        help='saved file name')
    parser.add_argument("--resize_to", type=int, default=256,
                        help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256,
                        help='crop the resized image to')
    parser.add_argument("--load_dataset", default=None,
                        help='load dataset')
    parser.add_argument("--category", default="A", type=str,
                        help="select A or B (A/B indicates trainA/B respectively")
    args = parser.parse_args()
    print(args)
    root = args.root

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    gen = getattr(net, args.gen_class)()

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator F model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        print("use gpu {}".format(args.gpu))

    if args.load_dataset is None:
        data_dir = root
    else:
        data_dir = os.path.join(root, args.load_dataset)
    data_dir = os.path.join(data_dir, "train{}".format(args.category.upper()))
    dataset = Dataset(path=data_dir, resize_to=args.resize_to,
                      crop_to=args.crop_to, flip=False)

    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,
                                                repeat=False, shuffle=False)

    xp = gen.xp
    cnt = 0
    for batch in iterator:
        imgs = chainer.dataset.concat_examples(batch, device=args.gpu)
        with chainer.using_config('train', False):
            out = xp.asnumpy(gen(imgs).data)
        for i in range(len(out)):
            path = '{:s}/{:s}.jpg'.format(args.out, dataset.ids[cnt])
            arr = (out[i] + 1.0) / 2.0 * 255.0
            org = read_image(dataset.get_img_path(cnt))
            _, h, w = org.shape
            arr = resize(arr, (h, w))
            write_image(arr, path)
            cnt += 1
            print(cnt)
