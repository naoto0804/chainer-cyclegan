#!/usr/bin/env python

import argparse

import chainer.cuda
import numpy as np
from chainer import serializers
from chainercv.transforms import resize
from chainercv.utils import read_image
from chainercv.utils import write_image

import net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image path')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', '-c', default='Generator',
                        help='Default gen erator class')
    parser.add_argument("--load_gen_model", '-l', default='',
                        help='load generator model')
    parser.add_argument('--output', '-o', default='result.jpg',
                        help='output image path')
    parser.add_argument("--base_size", '-s', type=int, default=256,
                        help='shorter edge length')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    gen = getattr(net, args.gen_class)()

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        print("use gpu {}".format(args.gpu))

    xp = gen.xp
    img = read_image(args.input)
    img = img.astype("f")
    img = img * 2 / 255.0 - 1.0  # [-1, 1)
    height, width = img.shape[1:]
    img = np.expand_dims(img, axis=0)
    img = xp.asarray(img)

    with chainer.using_config('train', False):
        out = gen(img)
    out = resize(xp.asnumpy(out.data[0]), (height, width))
    out = (out + 1.0) / 2.0 * 255.0

    write_image(out, args.output)
