#!/usr/bin/env python3

import argparse
import os

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import net as net
from dataset import Dataset
from visualization import visualize
from updater import Updater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--max_iter', '-m', type=int, default=120000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result')

    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002,
                        help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002,
                        help="Learning rate for discriminator")

    parser.add_argument("--load_gen_f_model", default='',
                        help='load generator model')
    parser.add_argument("--load_gen_g_model", default='',
                        help='load generator model')
    parser.add_argument("--load_dis_x_model", default='',
                        help='load discriminator model')
    parser.add_argument("--load_dis_y_model", default='',
                        help='load discriminator model')

    parser.add_argument('--gen_class', default='Generator',
                        help='Default generator class')
    parser.add_argument('--dis_class', default='Discriminator',
                        help='Default discriminator class')

    parser.add_argument("--lambda1", type=float, default=10.0,
                        help='lambda for reconstruction loss')
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help='lambda for adversarial loss')

    parser.add_argument("--flip", type=int, default=1,
                        help='flip images for data augmentation')
    parser.add_argument("--resize_to", type=int, default=286,
                        help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256,
                        help='crop the resized image to')
    parser.add_argument("--load_dataset", default='horse2zebra',
                        help='load dataset')
    parser.add_argument("--discriminator_layer_n", type=int, default=5,
                        help='number of discriminator layers')

    parser.add_argument("--lrdecay_start", type=float, default=100,
                        help='anneal the learning rate (by epoch)')
    parser.add_argument("--lrdecay_period", type=int,
                        default=100, help='period to anneal the learning')

    args = parser.parse_args()
    print(args)

    root = args.root
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    gen_g = getattr(net, args.gen_class)()
    dis_x = getattr(net, args.dis_class)()
    gen_f = getattr(net, args.gen_class)()
    dis_y = getattr(net, args.dis_class)()

    if args.load_gen_g_model != '':
        serializers.load_npz(args.load_gen_g_model, gen_g)
        print("Generator G(X->Y) model loaded")

    if args.load_gen_f_model != '':
        serializers.load_npz(args.load_gen_f_model, gen_f)
        print("Generator F(Y->X) model loaded")

    if args.load_dis_x_model != '':
        serializers.load_npz(args.load_dis_x_model, dis_x)
        print("Discriminator X model loaded")

    if args.load_dis_y_model != '':
        serializers.load_npz(args.load_dis_y_model, dis_y)
        print("Discriminator Y model loaded")

    if not os.path.exists(args.eval_folder):
        os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0:
        gen_g.to_gpu()
        gen_f.to_gpu()
        dis_x.to_gpu()
        dis_y.to_gpu()
        print("use gpu {}".format(args.gpu))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    opt_g = make_optimizer(gen_g, alpha=args.learning_rate_g)
    opt_f = make_optimizer(gen_f, alpha=args.learning_rate_g)
    opt_x = make_optimizer(dis_x, alpha=args.learning_rate_d)
    opt_y = make_optimizer(dis_y, alpha=args.learning_rate_d)

    train_A_dataset = Dataset(
        path=os.path.join(root, args.load_dataset, 'trainA'), flip=args.flip,
        resize_to=args.resize_to, crop_to=args.crop_to)
    train_B_dataset = Dataset(
        path=os.path.join(root, args.load_dataset, 'trainB'), flip=args.flip,
        resize_to=args.resize_to, crop_to=args.crop_to)

    # train_A_iter = chainer.iterators.MultiprocessIterator(
    #     train_A_dataset, args.batch_size, n_processes=4)
    # train_B_iter = chainer.iterators.MultiprocessIterator(
    #     train_B_dataset, args.batch_size, n_processes=4)

    train_A_iter = chainer.iterators.SerialIterator(
        train_A_dataset, args.batch_size)
    train_B_iter = chainer.iterators.SerialIterator(
        train_B_dataset, args.batch_size)

    # Set up a trainer
    updater = Updater(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_A_iter,
            'train_B': train_B_iter,
        },
        optimizer={
            'gen_g': opt_g,
            'gen_f': opt_f,
            'dis_x': opt_x,
            'dis_y': opt_y
        },
        device=args.gpu,
        params={
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'image_size': args.crop_to,
            'eval_folder': args.eval_folder,
            'lrdecay_start': args.lrdecay_start,
            'lrdecay_period': args.lrdecay_period,
            'dataset': train_A_dataset
        })

    log_interval = (20, 'iteration')
    model_save_interval = (5000, 'iteration')
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=args.out)
    trainer.extend(extensions.snapshot_object(
        gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        gen_f, 'gen_f{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_x, 'dis_x{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_y, 'dis_y{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen_g/loss_rec', 'gen_f/loss_rec',
                'gen_g/loss_gen', 'gen_f/loss_gen', 'dis_x/loss', 'dis_y/loss']
    trainer.extend(
        extensions.LogReport(keys=log_keys, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['gen_g/loss_rec', 'gen_f/loss_rec', 'gen_g/loss_gen',
                 'gen_f/loss_gen', 'dis_x/loss', 'dis_y/loss'], 'iteration',
                trigger=(100, 'iteration'), file_name='loss.png'))

    trainer.extend(
        visualize(gen_g, gen_f, args.eval_folder),
        trigger=(1, 'epoch')
    )

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
