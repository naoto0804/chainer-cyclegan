import os

import chainer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from chainer import Variable


def postprocess(var):
    img = var.data.get()
    img = (img + 1.0) / 2.0  # [0, 1)
    return img.transpose(0, 2, 3, 1)


def visualize(gen_g, gen_f, test_image_folder):
    @chainer.training.make_extension()
    def visualization(trainer):
        updater = trainer.updater
        batch_x = updater.get_iterator('main').next()
        batch_y = updater.get_iterator('train_B').next()
        batchsize = len(batch_x)
        fig = plt.figure(figsize=(3, 2 * batchsize))
        gs = gridspec.GridSpec(2 * batchsize, 3, wspace=0.1, hspace=0.1)

        x = Variable(updater.converter(batch_x, updater.device))
        y = Variable(updater.converter(batch_y, updater.device))

        with chainer.using_config('train', False):
            x_y = gen_g(x)
            x_y_x = gen_f(x_y)

        for i, var in enumerate([x, x_y, x_y_x]):
            imgs = postprocess(var)
            for j in range(batchsize):
                ax = fig.add_subplot(gs[j * 2, i])
                ax.imshow(imgs[j], interpolation='none')
                ax.set_xticks([])
                ax.set_yticks([])

        with chainer.using_config('train', False):
            y_x = gen_f(y)
            y_x_y = gen_g(y_x)

        for i, var in enumerate([y, y_x, y_x_y]):
            imgs = postprocess(var)
            for j in range(batchsize):
                ax = fig.add_subplot(gs[j * 2 + 1, i])
                ax.imshow(imgs[j], interpolation='none')
                ax.set_xticks([])
                ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(test_image_folder,
                                 'epoch{:d}.jpg'.format(updater.epoch)))

    return visualization
