import sys

sys.path.append('..')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataProcess.data_parser import *
import os
import cv2
from ColorModel.vgg19 import Vgg19


def get_feature_map(image, reuse=False):
    vgg = Vgg19()
    with tf.variable_scope('vgg', reuse):
        img_resize = tf.image.resize_images(image, (224, 224), 2)
        vgg.build(img_resize)
    return vgg.fc8


class Unet:
    def __init__(self, image_shape, batch_size, train_strategy):
        """
        currently our model contains an unet-like generator and a simple discriminator,
        support training with GAN, WGAN-GP and non-adversarial method
        :param image_shape: shape of image
        :param batch_size: batch size
        :param train_strategy: must contained in ['unet', 'gan', 'wgangp']
        """
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.train_strategy = train_strategy

    def generator(self, input_img, condition=None, is_train=True):
        """
        an unet-like structure
        :param input_img: place_holder of input image
        :param condition: whether adding condition to this model
        :param is_train: parameter of batch norm layer
        :return: a 3 channel image with sigmoid active function
        """
        if not condition:
            inputs = input_img
        else:
            inputs = tf.concat(values=[input_img, condition], axis=3)
        with tf.variable_scope('generator'):
            encode1, sample_encode1 = self._cnn_block(inputs, 64, (3, 3), sample_type='conv', scope_name='encode1',
                                                      is_train=is_train)
            encode2, sample_encode2 = self._cnn_block(sample_encode1, 128, (3, 3), sample_type='conv',
                                                      scope_name='encode2', is_train=is_train)
            encode3, sample_encode3 = self._cnn_block(sample_encode2, 256, (3, 3), sample_type='conv',
                                                      scope_name='encode3', is_train=is_train)
            encode4, sample_encode4 = self._cnn_block(sample_encode3, 512, (3, 3), sample_type='conv',
                                                      scope_name='encode4', is_train=is_train)

            with tf.variable_scope('last_encode'):
                layer = tf.layers.conv2d(inputs=sample_encode4, filters=1024, kernel_size=(3, 3), name='conv',
                                         padding='same')
                layer = tf.layers.batch_normalization(layer, training=is_train)
                layer = tf.nn.leaky_relu(layer)
            decode1, _ = self._cnn_block(layer, 512, (3, 3), sample_type='deconv', scope_name='decode1',
                                         deconv_concatenate=encode4, is_train=is_train)
            decode2, _ = self._cnn_block(decode1, 256, (3, 3), sample_type='deconv', scope_name='decode2',
                                         deconv_concatenate=encode3, is_train=is_train)
            decode3, _ = self._cnn_block(decode2, 128, (3, 3), sample_type='deconv', scope_name='decode3',
                                         deconv_concatenate=encode2, is_train=is_train)
            decode4, _ = self._cnn_block(decode3, 64, (3, 3), sample_type='deconv', scope_name='decode4',
                                         deconv_concatenate=encode1, is_train=is_train)

            g_logits = tf.layers.conv2d(decode4, 3, (3, 3), padding='same', name='logits', activation=tf.nn.sigmoid)

        return g_logits

    def discriminator(self, input_img, condition=None, reuse=False):
        """
        simple discriminator, need to change to patchCNN later
        :param input_img: fake or real image
        :param condition: if given, it would be a cGAN model
        :param reuse: used for distinguish real and fake
        :return: scalar 0(fake)/1(real)
        """
        if not condition:
            inputs = input_img
        else:
            inputs = tf.concat(values=[input_img, condition], axis=3)
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = tf.layers.conv2d(inputs, 64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.leaky_relu,
                                     name='conv1')
            # layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool1')
            layer = tf.layers.conv2d(layer, 128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.leaky_relu,
                                     name='conv2')
            # layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool2')
            layer = tf.layers.conv2d(layer, 128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.leaky_relu,
                                     name='conv3')
            # layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool3')
            layer = tf.layers.conv2d(layer, 64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.leaky_relu,
                                     name='conv4')
            # layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool4')
            layer = tf.layers.flatten(layer)
            d_logits = tf.layers.dense(layer, 1000, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.relu)
            d_logits = tf.layers.dense(d_logits, 1,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        if self.train_strategy == 'wgangp':
            return d_logits
        return tf.nn.sigmoid(d_logits)

    def _cnn_block(self, input_mat, num_filter, kernel_size, sample_type, scope_name, is_train, deconv_concatenate=None,
                   reuse=False):
        if sample_type not in ['conv', 'deconv']:
            raise ValueError('Undefined sample type')
        with tf.variable_scope(scope_name, reuse=reuse):
            if sample_type == 'conv':
                # cnn1 = tf.layers.conv2d(inputs=input, filters=num_filter, kernel_size=kernel_size, padding='same',
                #                         name='conv1',
                #                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                # cnn1 = tf.layers.batch_normalization(cnn1, training=is_train)
                # cnn1 = tf.nn.leaky_relu(cnn1)
                out = tf.layers.conv2d(inputs=input_mat, filters=num_filter, kernel_size=kernel_size, padding='same',
                                       name='conv2',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                out = tf.layers.batch_normalization(out, training=is_train)
                out = tf.nn.leaky_relu(out)
                sample_out = tf.layers.conv2d(inputs=out, filters=num_filter * 2, kernel_size=kernel_size,
                                              strides=(2, 2),
                                              padding='same',
                                              name='downsampling',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                sample_out = tf.layers.batch_normalization(sample_out, training=is_train)
                sample_out = tf.nn.leaky_relu(sample_out)

            else:
                if deconv_concatenate is None:
                    raise ValueError('deconv_concatenate can not be None when building deconv structure')
                dcnn = tf.layers.conv2d_transpose(inputs=input_mat, filters=num_filter, kernel_size=kernel_size,
                                                  strides=2,
                                                  padding='same', name='deconv1',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                dcnn = tf.layers.batch_normalization(dcnn, training=is_train)
                dcnn = tf.nn.leaky_relu(dcnn)
                concat = tf.concat(values=[dcnn, deconv_concatenate], axis=3)
                # dcnn1 = tf.layers.conv2d(inputs=concat, filters=num_filter, kernel_size=kernel_size, padding='same',
                #                          name='d_conv1',
                #                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                # dcnn1 = tf.layers.batch_normalization(dcnn1, training=is_train)
                # dcnn1 = tf.nn.leaky_relu(dcnn1)
                out = tf.layers.conv2d(inputs=concat, filters=num_filter, kernel_size=kernel_size, padding='same',
                                       name='d_conv2',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                out = tf.layers.batch_normalization(out, training=is_train)
                out = tf.nn.leaky_relu(out)
                sample_out = None
        return out, sample_out

    def _get_feed_dict(self, train_strategy, input_img, condition, target, is_training):
        feed_dict = {}
        feed_dict[self.input_img] = input_img
        if condition:
            feed_dict[self.condition] = condition
        feed_dict[self.target] = target
        feed_dict[self.is_training] = is_training
        if train_strategy == 'wgangp':
            e = np.random.uniform(0, 1, [target.shape[0], 1, 1, 1])
            feed_dict[self.random_e] = e
        return feed_dict

    def train(self, data_parser, epochs, model_save_path,
              val_batch, learning_rate_g=0.0001, learning_rate_d=0.0002,
              it_d=1, it_g=1,
              use_l1=False, use_l2=False, use_condition=True,
              l1_rate=0.001, l2_rate=0.01, penalty_rate=10,
              save_index=1, restore_path=None, resume_path=None, restart=0,
              verbose=True, loss_per_iteration=True, save_epoch_gap=10,
              max_ckpt=100):
        """
        notice we can only train this model once after initialization
        form of losses:
        0: d_loss
        1: expectation_fake
        2: expectation_real
        3: g_loss
        4: l1_loss
        5: l2_loss
        6: d_loss_val
        7: expectation_fake_val
        8: expectation_real_val
        9: g_loss_val
        10: l1_loss_val
        11: l2_loss_val

        :param data_parser: DataParserV2 or DataParserV3
        :param epochs: training epochs (not iteration)
        :param model_save_path: path to save model and logs, we'll automatically create logs/ and checkpoints/, where
        logs/ stores the losses and generated images, checkpoints/ stores the middle and final models
        :param val_batch: validation data, should be a dictionary which contains 'input', 'condition' and 'target'
        :param learning_rate_g: learning rate of generator
        :param learning_rate_d: learning rate of discriminator
        :param it_d: training times of discriminator per iteration
        :param it_g: training times of generator per iteration
        :param use_l1: whether to add l1 loss (pixel-level) to generator loss
        :param use_l2: whether to add l2 loss (feature-level using Vgg19) to generator loss
        :param use_condition: whether to use condition (color hint in this project) for training
        :param l1_rate: ratio of l1 loss
        :param l2_rate: ratio of l2 loss
        :param penalty_rate: ratio of gradient penalty, used in WGAN-GP
        :param save_index: index of current model
        :param restore_path: path of *.ckpt file, if given, restore generator's parameters before training
        :param resume_path: path of *.ckpt file, if given, resume training from last saved checkpoint
        :param restart: index of epoch to resume training
        :param verbose: show training information of not
        :param loss_per_iteration: save losses data every iteration, if False, save them every epoch
        :param save_epoch_gap: gap to save model, losses and generated results
        :param max_ckpt: max number of checkpoints to store
        :return: None
        """
        tf.reset_default_graph()

        self.input_img = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 1),
                                        name='input')
        if use_condition:
            self.condition = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                            name='condition')
        else:
            self.condition = None
        self.target = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                     name='output')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.random_e = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='random_e')

        g_logits = self.generator(self.input_img, self.condition, self.is_training)
        tf.add_to_collection('g_logits', g_logits)

        if use_l1:
            l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g_logits - self.target), axis=[1, 2, 3]))
        else:
            l1_loss = 0

        if use_l2:
            feature_fake = get_feature_map(g_logits)
            feature_real = get_feature_map(self.target, reuse=True)
            l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(feature_fake - feature_real), axis=1))
        else:
            l2_loss = 0

        eps = 1e-8
        if self.train_strategy == 'unet':
            g_loss = l1_rate * l1_loss + l2_rate * l2_loss
            use_discriminator = False
        else:
            use_discriminator = True

            d_logits_real = self.discriminator(self.target, self.condition, reuse=False)
            d_logits_fake = self.discriminator(g_logits, self.condition, reuse=True)
            expect_real = tf.reduce_mean(d_logits_real)
            expect_fake = tf.reduce_mean(d_logits_fake)
            tf.add_to_collection('d_logits_real', d_logits_real)
            tf.add_to_collection('d_logits_fake', d_logits_fake)
            tf.add_to_collection('expect_real', expect_real)
            tf.add_to_collection('expect_fake', expect_fake)

            if self.train_strategy == 'gan':
                g_loss = -tf.reduce_mean(tf.log(d_logits_fake + eps)) + l1_rate * l1_loss + l2_rate * l2_loss
                d_loss = -tf.reduce_mean(tf.log(d_logits_real + eps) + tf.log(1 - d_logits_fake + eps))
            elif self.train_strategy == 'wgangp':
                g_loss = -tf.reduce_mean(d_logits_fake + eps) + l1_rate * l1_loss + l2_rate * l2_loss

                x_hat = self.random_e * self.target + (1 - self.random_e) * g_logits
                d_logits_xhat = self.discriminator(x_hat, self.condition, reuse=True)
                grads = tf.gradients(d_logits_xhat, [x_hat])[0]
                penalty = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + eps) - 1))
                d_loss = tf.reduce_mean(d_logits_fake - d_logits_real) + penalty_rate * penalty
            else:
                raise ValueError('Undefined training strategy, please wait for further update')

        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        bn_ops_g = [var for var in bn_ops if var.name.startswith('generator')]
        var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        with tf.control_dependencies(bn_ops_g):
            g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_g).minimize(g_loss, var_list=var_g)

        if use_discriminator:
            bn_ops_d = [var for var in bn_ops if var.name.startswith('discriminator')]
            var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
            with tf.control_dependencies(bn_ops_d):
                d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(d_loss, var_list=var_d)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        losses = []

        model_dir = os.path.join(model_save_path, 'model_{}({})/checkpoints'.format(save_index, self.train_strategy))
        log_dir = os.path.join(model_save_path, 'model_{}/({})logs'.format(save_index, self.train_strategysbl123))
        try:
            os.makedirs(model_dir)
        except FileExistsError as e:
            print(e)
        try:
            os.makedirs(log_dir)
        except FileExistsError as e:
            print(e)

        restore_list = tf.trainable_variables('generator')
        restore_saver = tf.train.Saver(restore_list)

        model_saver = tf.train.Saver(max_to_keep=max_ckpt)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if restore_path:
                restore_saver.restore(sess, restore_path)
            elif resume_path:
                model_saver.restore(sess, resume_path)
            loss_l1 = 0
            loss_l1_val = 0
            loss_l2 = 0
            loss_l2_val = 0
            loss_g = 0
            loss_g_val = 0
            loss_d = 0
            loss_d_val = 0
            fake_exp = 0
            fake_exp_val = 0
            real_exp = 0
            real_exp_val = 0
            for epoch in range(restart, epochs):
                for i in range(data_parser.iteration):
                    batch_input = data_parser.get_batch_sketch()
                    batch_target = data_parser.get_batch_raw()
                    batch_condition = data_parser.get_batch_condition_add()
                    data_parser.update_iterator()

                    feed_dict = self._get_feed_dict(self.train_strategy, batch_input, batch_condition, batch_target,
                                                    True)
                    if use_discriminator:
                        for _ in range(it_d):
                            _, loss_d, fake_exp, real_exp = sess.run([d_optimizer, d_loss, expect_fake, expect_real],
                                                                     feed_dict=feed_dict, options=run_options)
                    for _ in range(it_g):
                        _, loss_g, loss_l1, loss_l2 = sess.run([g_optimizer, g_loss, l1_loss, l2_loss],
                                                               feed_dict=feed_dict, options=run_options)
                    feed_dict_val = self._get_feed_dict(self.train_strategy, val_batch['input'], val_batch['condition'],
                                                        val_batch['target'], False)

                    loss_d_val, fake_exp_val, real_exp_val, loss_g_val, loss_l1_val, loss_l2_val = sess.run(
                        [d_loss, expect_fake, expect_real, g_loss, l1_loss, l2_loss], feed_dict=feed_dict_val)
                    if loss_per_iteration:
                        losses.append([loss_d, fake_exp, real_exp, loss_g, loss_l1, loss_l2,
                                       loss_d_val, fake_exp_val, real_exp_val, loss_g_val, loss_l1_val, loss_l2_val])
                    if i % 10 == 0 and verbose:
                        print('Epoch: {}, Iteration: {}/{} ...'.format(epoch + 1, i + 1, data_parser.iteration),
                              'g_loss: {:.4f}, l1_loss: {:.4f}, l2_loss: {:.4f} ...'.format(loss_g, loss_l1, loss_l2),
                              'd_loss: {:.4f}, exp_real: {:.4f}, exp_fake: {:.4f} ...'.format(loss_d, real_exp,
                                                                                              fake_exp),
                              'g_loss_val: {:.4f}, l1_loss_val: {:.4f}, l2_loss_val: {:.4f} ...'.format(loss_g_val,
                                                                                                        loss_l1_val,
                                                                                                        loss_l2_val),
                              'd_loss_val: {:.4f}, exp_real_val: {:.4f}, exp_fake_val: {:.4f} ...'.format(loss_d_val,
                                                                                                          real_exp_val,
                                                                                                          fake_exp_val)
                              )
                if verbose:
                    print('*' * 10,
                          'Epoch: {}/{} ...'.format(epoch + 1, epochs),
                          'g_loss: {:.4f}, l1_loss: {:.4f}, l2_loss: {:.4f} ...'.format(loss_g, loss_l1, loss_l2),
                          'd_loss: {:.4f}, exp_real: {:.4f}, exp_fake: {:.4f} ...'.format(loss_d, real_exp,
                                                                                          fake_exp),
                          'g_loss_val: {:.4f}, l1_loss_val: {:.4f}, l2_loss_val: {:.4f} ...'.format(loss_g_val,
                                                                                                    loss_l1_val,
                                                                                                    loss_l2_val),
                          'd_loss_val: {:.4f}, exp_real_val: {:.4f}, exp_fake_val: {:.4f} ...'.format(loss_d_val,
                                                                                                      real_exp_val,
                                                                                                      fake_exp_val),
                          '*' * 10
                          )
                if not loss_per_iteration:
                    losses.append([loss_d, fake_exp, real_exp, loss_g, loss_l1, loss_l2,
                                   loss_d_val, fake_exp_val, real_exp_val, loss_g_val, loss_l1_val, loss_l2_val])

                if (epoch + 1) % save_epoch_gap == 0:
                    print('*' * 10, 'save results', '*' * 10)
                    feed_dict_pred = self._get_feed_dict(self.train_strategy, val_batch['input'],
                                                         val_batch['condition'],
                                                         val_batch['target'], False)
                    pred = sess.run(g_logits, feed_dict=feed_dict_pred)
                    np.save(os.path.join(log_dir, 'predict-{}'.format(epoch + 1)), pred)
                    np.save(os.path.join(log_dir, 'losses-{}'.format(epoch + 1)), np.asarray(losses))
                    losses = []
                    print('*' * 10, 'save checkpoint', '*' * 10)
                    model_saver.save(sess, os.path.join(model_dir, 'checkpoint-{}.ckpt'.format(epoch + 1)))
            print('*' * 10, 'save model', '*' * 10)
            model_saver.save(sess, os.path.join(model_dir, 'model-{}.ckpt'.format(epochs)))

    def generate(self, model_dir, model_name, input_img, cond=None):
        """
        generate fake image through trained model
        :param model_dir: directory of model
        :param model_name: name of model/checkpoint
        :param input_img: input sketch, must has shape (x, x, x, x)
        :param cond: condition (color hint), same as input_img
        :return: generated images
        """
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_dir, model_name + '.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name('input:0')
        if cond:
            condition = graph.get_tensor_by_name('condition:0')
        is_train = graph.get_tensor_by_name('is_training:0')
        g_logits = graph.get_collection('g_logits')

        feed_dict = {}
        feed_dict[inputs] = input_img
        if cond:
            feed_dict[condition] = cond
        feed_dict[is_train] = False
        generated = sess.run(g_logits, feed_dict=feed_dict)
        sess.close()
        return generated


if __name__ == '__main__':
    # usage:
    img_shape = (256, 256)
    batch_size = 10
    train_strategy = 'wgangp'
    model_save_path = 'xxx'
    epochs = 100

    dataset_path = 'xxx'
    list_files = ['../dataset/image_list_1.txt']
    data_parser = DataParserV2(dataset_path, img_shape, list_files, batch_size)

    val_path = 'test.npy'
    val_batch = np.load(val_path).item()

    model = Unet(img_shape, batch_size, train_strategy)
    model.train(data_parser, epochs=epochs, model_save_path=model_save_path, val_batch=val_batch)
