import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._lrdecay_start = params['lrdecay_start']
        self._lrdecay_period = params['lrdecay_period']
        self._image_size = params['image_size']
        self._eval_foler = params['eval_folder']
        self._dataset = params['dataset']
        self._iter = 0
        self._max_buffer_size = 50
        self.xp = self.gen_g.xp
        self._buffer_x = self.xp.zeros(
            (self._max_buffer_size, 3, self._image_size,
             self._image_size)).astype("f")
        self._buffer_y = self.xp.zeros(
            (self._max_buffer_size, 3, self._image_size,
             self._image_size)).astype("f")
        self.init_alpha = self.get_optimizer('gen_g').alpha

    def getAndUpdateBufferX(self, data):
        if self._iter < self._max_buffer_size:
            self._buffer_x[self._iter, :] = data[0]
            return data

        self._buffer_x[0:self._max_buffer_size - 2, :] = self._buffer_x[
                                                         1:self._max_buffer_size - 1,
                                                         :]
        self._buffer_x[self._max_buffer_size - 1, :] = data[0]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, self._max_buffer_size)
        return self._buffer_x[id, :].reshape(
            (1, 3, self._image_size, self._image_size))

    def getAndUpdateBufferY(self, data):

        if self._iter < self._max_buffer_size:
            self._buffer_y[self._iter, :] = data[0]
            return data

        self._buffer_y[0:self._max_buffer_size - 2, :] = self._buffer_y[
                                                         1:self._max_buffer_size - 1,
                                                         :]
        self._buffer_y[self._max_buffer_size - 1, :] = data[0]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, self._max_buffer_size)
        return self._buffer_y[id, :].reshape(
            (1, 3, self._image_size, self._image_size))

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

    def update_learning_schedule(self):
        pass

    def update_core(self):
        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')
        self._iter += 1
        if self.is_new_epoch and self.epoch >= self._lrdecay_start:
            decay_step = self.init_alpha / self._lrdecay_period
            print("lr decay", decay_step)
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

        w_in = self._image_size
        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        # TODO If we use update buffer, it is more difficult to add task specific loss?

        x_y = self.gen_g(x)
        x_y_copy = self.getAndUpdateBufferX(x_y.data)
        x_y_copy = Variable(x_y_copy)
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = self.getAndUpdateBufferY(y_x.data)
        y_x_copy = Variable(y_x_copy)
        y_x_y = self.gen_g(y_x)

        loss_gen_g_adv = self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = self.loss_func_adv_gen(self.dis_x(y_x))

        loss_cycle_x = self._lambda1 * self.loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda1 * self.loss_func_rec_l1(y_x_y, y)
        loss_gen = self._lambda2 * loss_gen_g_adv + \
                   self._lambda2 * loss_gen_f_adv + \
                   loss_cycle_x + loss_cycle_y
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
        chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_gen': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_gen': loss_gen_f_adv}, self.gen_f)
