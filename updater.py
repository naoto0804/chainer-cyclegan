import random

import chainer
import chainer.functions as F
from chainer import Variable


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        xp = chainer.cuda.get_array_module(images)
        for image in images:
            image = xp.expand_dims(image, axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = xp.copy(self.images[random_id])
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = xp.concatenate(return_images)
        return return_images


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self._lambda_A = params['lambda_A']
        self._lambda_B = params['lambda_B']
        self._lambda_id = params['lambda_identity']
        self._lrdecay_start = params['lrdecay_start']
        self._lrdecay_period = params['lrdecay_period']
        self._image_size = params['image_size']
        self._dataset = params['dataset']
        self._batch_size = params['batch_size']
        self._iter = 0
        self.xp = self.gen_g.xp
        self._buffer_x = ImagePool(50 * self._batch_size)
        self._buffer_y = ImagePool(50 * self._batch_size)
        self.init_alpha = self.get_optimizer('gen_g').alpha

    def loss_func_rec_l1(self, x_out, t):
        return F.mean_absolute_error(x_out, t)

    def loss_func_adv_dis_fake(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 0.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def loss_func_adv_dis_real(self, y_real):
        target = Variable(
            self.xp.full(y_real.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_real, target)

    def loss_func_adv_gen(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def update_core(self):
        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')
        self._iter += 1
        if self.is_new_epoch and self.epoch >= self._lrdecay_start:
            decay_step = self.init_alpha / self._lrdecay_period
            print('lr decay', decay_step)
            if opt_g.alpha > decay_step:
                opt_g.alpha -= decay_step
            if opt_f.alpha > decay_step:
                opt_f.alpha -= decay_step
            if opt_x.alpha > decay_step:
                opt_x.alpha -= decay_step
            if opt_y.alpha > decay_step:
                opt_y.alpha -= decay_step
        batch_x = self.get_iterator('main').next()
        batch_y = self.get_iterator('train_B').next()

        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        x_y = self.gen_g(x)
        x_y_copy = Variable(self._buffer_y.query(x_y.data))
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = Variable(self._buffer_x.query(y_x.data))
        y_x_y = self.gen_g(y_x)

        loss_gen_g_adv = self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = self.loss_func_adv_gen(self.dis_x(y_x))

        loss_cycle_x = self._lambda_A * self.loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda_B * self.loss_func_rec_l1(y_x_y, y)
        loss_id_x = self._lambda_id * F.mean_absolute_error(x, self.gen_f(x))
        loss_id_y = self._lambda_id * F.mean_absolute_error(y, self.gen_g(y))
        loss_gen = loss_gen_g_adv + loss_gen_f_adv + loss_cycle_x + \
                   loss_cycle_y + loss_id_x + loss_id_y

        self.gen_f.cleargrads()
        self.gen_g.cleargrads()
        loss_gen.backward()
        opt_f.update()
        opt_g.update()

        loss_dis_y_fake = self.loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = self.loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
        self.dis_y.cleargrads()
        loss_dis_y.backward()
        opt_y.update()

        loss_dis_x_fake = self.loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = self.loss_func_adv_dis_real(self.dis_x(x))
        loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
        self.dis_x.cleargrads()
        loss_dis_x.backward()
        opt_x.update()

        chainer.report({'loss': loss_dis_x}, self.dis_x)
        chainer.report({'loss': loss_dis_y}, self.dis_y)
        chainer.report({'loss_cycle': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_cycle': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_id': loss_id_y}, self.gen_g)
        chainer.report({'loss_id': loss_id_x}, self.gen_f)
        chainer.report({'loss_gen': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_gen': loss_gen_f_adv}, self.gen_f)
