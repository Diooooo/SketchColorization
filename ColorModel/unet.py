import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataProcess.data_parser import *
import os
import cv2


# base line
class Unet:
    def __init__(self, image_shape, batch_size):
        self.image_shape = image_shape
        self.batch_size = batch_size

    def generator(self, input, condition, is_train):
        inputs = tf.concat(values=[input, condition], axis=3)
        with tf.variable_scope('generator'):
            encode1, sample_encode1 = self.cnn_block(inputs, 64, (3, 3), sample_type='conv', scope_name='encode1',
                                                     is_train=is_train)
            encode2, sample_encode2 = self.cnn_block(sample_encode1, 128, (3, 3), sample_type='conv',
                                                     scope_name='encode2', is_train=is_train)
            encode3, sample_encode3 = self.cnn_block(sample_encode2, 256, (3, 3), sample_type='conv',
                                                     scope_name='encode3', is_train=is_train)
            encode4, sample_encode4 = self.cnn_block(sample_encode3, 512, (3, 3), sample_type='conv',
                                                     scope_name='encode4', is_train=is_train)

            with tf.variable_scope('last_encode'):
                layer = tf.layers.conv2d(inputs=sample_encode4, filters=1024, kernel_size=(3, 3), name='conv',
                                         padding='same')
                layer = tf.layers.batch_normalization(layer, training=is_train)
                layer = tf.nn.leaky_relu(layer)
            decode1, _ = self.cnn_block(layer, 512, (3, 3), sample_type='deconv', scope_name='decode1',
                                        deconv_concatenate=encode4, is_train=is_train)
            decode2, _ = self.cnn_block(decode1, 256, (3, 3), sample_type='deconv', scope_name='decode2',
                                        deconv_concatenate=encode3, is_train=is_train)
            decode3, _ = self.cnn_block(decode2, 128, (3, 3), sample_type='deconv', scope_name='decode3',
                                        deconv_concatenate=encode2, is_train=is_train)
            decode4, _ = self.cnn_block(decode3, 64, (3, 3), sample_type='deconv', scope_name='decode4',
                                        deconv_concatenate=encode1, is_train=is_train)

            g_logits = tf.layers.conv2d(decode4, 3, (3, 3), padding='same', name='logits', activation=tf.nn.sigmoid)

            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=g_logits)
            # cost = tf.reduce_mean(loss)
            # optimizer = tf.train.AdamOptimizer().minimize(cost)
        return g_logits

    def discriminator(self, input, condition, reuse=False):
        inputs = tf.concat(values=[input, condition], axis=3)
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
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.sigmoid)
        return d_logits

    def cnn_block(self, input, num_filter, kernel_size, sample_type, scope_name, is_train, deconv_concatenate=None,
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
                out = tf.layers.conv2d(inputs=input, filters=num_filter, kernel_size=kernel_size, padding='same',
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
                dcnn = tf.layers.conv2d_transpose(inputs=input, filters=num_filter, kernel_size=kernel_size, strides=2,
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

    def train(self, dataset_path, list_files, epochs, epochs_pre=20, learning_rate=0.01, clip_low=-0.01, clip_high=0.01,
              r=1, l=10,
              save_index=1):
        tf.reset_default_graph()
        input = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 1), name='input')
        condition = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                   name='condition')
        target = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3), name='output')
        is_training = tf.placeholder(tf.bool, name='is_training')
        # random_e = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='random_e')

        g_logits = self.generator(input, condition, is_training)

        d_logits_real = self.discriminator(target, condition, reuse=False)
        d_logits_fake = self.discriminator(g_logits, condition, reuse=True)

        # l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g_logits - target), axis=[1, 2, 3]))
        # l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(g_logits - target), axis=[1, 2, 3]))
        # print(l1_loss.shape)
        # g_loss = -tf.reduce_mean(d_logits_fake)
        eps = 1e-16
        g_loss = -tf.reduce_mean(tf.log(d_logits_fake + eps))
        # x_hat = random_e * target + (1 - random_e) * g_logits
        # d_logits_xhat = self.discriminator(x_hat, condition, reuse=True)
        # grads = tf.gradients(d_logits_xhat, [x_hat])[0]
        # penalty = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + eps) - 1))
        # d_loss = tf.reduce_mean(d_logits_fake - d_logits_real) + l * penalty
        d_loss = -tf.reduce_mean(tf.log(d_logits_real + eps) + tf.log(1 - d_logits_fake + eps))
        # tf.summary.scalar('g_loss', g_loss)
        # tf.summary.scalar('d_loss', d_loss)

        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        bn_ops_g = [var for var in bn_ops if var.name.startswith('generator')]
        bn_ops_d = [var for var in bn_ops if var.name.startswith('discriminator')]
        var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        with tf.control_dependencies(bn_ops_g):
            g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=var_g)
        with tf.control_dependencies(bn_ops_d):
            d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=var_d)
        # g_pretrain = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l2_loss, var_list=var_g)

        # clip_d_var = [var.assign(tf.clip_by_value(var, clip_low, clip_high)) for var in var_d]

        data_parser = DataParserV2(dataset_path, self.image_shape, list_files=list_files, batch_size=self.batch_size)
        data_parser_test = DataParserV2(dataset_path, self.image_shape, list_files=['../dataset/image_list_8.txt'],
                                        batch_size=20)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        losses = []
        losses_pretrain = []

        test_input = data_parser_test.get_batch_sketch()
        test_target = data_parser_test.get_batch_raw()
        test_condition = data_parser_test.get_batch_condition_add()
        test = {'input': test_input, 'target': test_target, 'condition': test_condition}
        np.save('./results/test_data_{}.npy'.format(save_index), test)

        # test = np.load('./results/test_data_{}.npy'.format(save_index)).item()
        outputs = []
        # outputs_pretrian = []
        try:
            os.makedirs('./checkpoints/unet{}'.format(save_index))
        except FileExistsError as e:
            print(e)
        var_list = tf.trainable_variables('generator')
        # bn_var = [var for var in tf.global_variables('generator') if 'moving_mean' in var.name]
        # bn_var += [var for var in tf.global_variables('generator') if 'moving_variance' in var.name]
        # var_list += bn_var
        saver = tf.train.Saver(var_list)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # init_var = [var for var in tf.global_variables() if 'generator' not in var.name]
        # for var in tf.global_variables():
        #     print(var)
        with tf.Session(config=config) as sess:
            # writer = tf.summary.FileWriter('./logs/train', sess.graph)
            # writer_val = tf.summary.FileWriter('./logs/val')

            # merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './checkpoints/unet_simple27/model.ckpt')

            # pre-train generator
            # for g_epoch in range(epochs_pre):
            #     for i in range(data_parser.iteration):
            #         batch_input = data_parser.get_batch_sketch()
            #         # print(batch_input.dtype, batch_input.shape)
            #         batch_target = data_parser.get_batch_raw()
            #         # print(batch_target.dtype, batch_target.shape)
            #         # batch_condition = data_parser.get_batch_condition_add()
            #         # print(batch_condition.dtype, batch_condition.shape)
            #         data_parser.update_iterator()
            #
            #         _, loss_l2 = sess.run([g_pretrain, l2_loss],
            #                               feed_dict={input: batch_input,
            #                                          target: batch_target, is_training: True},
            #                               options=run_options)
            #         losses_pretrain.append(loss_l2)
            #         if i % 10 == 0:
            #             print(
            #                 'Epoch: {}, Iteration: {}/{}, l1_loss: {}'.format(
            #                     g_epoch + 1, i + 1, data_parser.iteration, loss_l2))
            #     print('*' * 10,
            #           'Epoch {}/{} ...'.format(g_epoch + 1, epochs_pre),
            #           'l1_loss: {:.4f} ...'.format(loss_l2),
            #           '*' * 10)
            #     output_pre = sess.run(g_logits, feed_dict={input: test_input,
            #                                                target: test_target, is_training: False})
            #     outputs_pretrian.append(output_pre)

            # for _ in range(10):
            #     print('*' * 10)

            for epoch in range(epochs):
                loss_g = 0
                loss_d = 0
                i_d = 1
                i_g = 1
                for i in range(data_parser.iteration):
                    batch_input = data_parser.get_batch_sketch()
                    batch_target = data_parser.get_batch_raw()
                    batch_condition = data_parser.get_batch_condition_add()
                    data_parser.update_iterator()

                    for _ in range(i_d):
                        # e = np.random.uniform(0, 1, [batch_target.shape[0], 1, 1, 1])
                        # e = tf.random_uniform(shape=[batch_target.shape[0], 1], minval=0, maxval=1)
                        _, loss_d = sess.run([d_optimizer, d_loss],
                                             feed_dict={input: batch_input, condition: batch_condition,
                                                        target: batch_target, is_training: True},
                                             options=run_options)
                        # sess.run(clip_d_var)

                    for _ in range(i_g):
                        _, loss_g = sess.run([g_optimizer, g_loss],
                                             feed_dict={input: batch_input, condition: batch_condition,
                                                        target: batch_target, is_training: True},
                                             options=run_options)
                    loss_d_val, loss_g_val = sess.run([d_loss, g_loss],
                                                      feed_dict={input: test['input'], condition: test['condition'],
                                                                 target: test['target'], is_training: False})
                    # log = sess.run(merged)
                    # writer.add_summary(log, epoch * data_parser_inputs.iteration + i + 1)
                    losses.append([loss_d, loss_g, loss_d_val, loss_g_val])
                    if i % 10 == 0:
                        # log_val, val_loss_g, val_loss_d = sess.run([merged, g_loss, d_loss],
                        #                                            feed_dict={input: batch_input,
                        #                                                       condition: batch_condition,
                        #                                                       target: batch_target})
                        # writer_val.add_summary(log_val, epoch * data_parser_inputs.iteration + i + 1)
                        print(
                            'Epoch: {}, Iteration: {}/{}, g_loss: {}, d_loss: {}'.format(
                                epoch + 1, i + 1, data_parser.iteration, loss_g, loss_d))
                print('*' * 10,
                      'Epoch {}/{} ...'.format(epoch + 1, epochs),
                      'g_loss: {:.4f} ...'.format(loss_g),
                      'd_loss: {:.4f} ...'.format(loss_d),
                      # 'l1_loss: {:.4f} ...'.format(loss_l1),
                      '*' * 10)
                if (epochs + 1) % 20 == 0:
                    output = sess.run(g_logits, feed_dict={input: test['input'], condition: test['condition'],
                                                           target: test['target'], is_training: False})
                    outputs.append(output)
                # saver.save(sess, './checkpoints/unet8/checkpoint_{}.ckpt'.format(epoch + 1))
            saver.save(sess, './checkpoints/unet{}/model.ckpt'.format(save_index))
            np.save('./results/predicted_{}.npy'.format(save_index), np.asarray(outputs))
            # np.save('./results/predicted_pre_{}.npy'.format(save_index), np.asarray(outputs_pretrian))
        return losses


if __name__ == '__main__':
    dataset_path = '/media/bilin/MyPassport/data/dataset'
    resize_shape = (256, 256)
    list_files = ['../dataset/image_list_1.txt']
    batch_size = 10
    l1_rate = 0.005
    index = 30

    model = Unet(resize_shape, batch_size=batch_size)

    losses = model.train(dataset_path, list_files, 100, learning_rate=0.0001,
                         save_index=index)
    losses = np.asarray(losses)
    np.save('./logs/train/losses_{}.npy'.format(index), losses)
    # np.save('./logs/train/losses_pre_{}.npy'.format(index), np.asarray(losses_pretrain))

    # losses = np.load('./logs/train/losses_{}.npy'.format(index))
    # losses_pretrain = np.load('./logs/train/losses_pre_{}.npy'.format(index))
    # losses_pretrain = np.asarray(losses_pretrain)
    plt.plot(losses[:, 0], label='train')
    plt.plot(losses[:, 2], label='val')
    plt.legend(loc='upper right')
    plt.title('d loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(losses[:, 1], label='train')
    plt.plot(losses[:, 3], label='val')
    plt.legend(loc='upper right')
    plt.title('g loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # plt.plot(losses_pretrain)
    # plt.title('l1 loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    test = np.load('./results/test_data_{}.npy'.format(index))
    imgs = np.load('./results/predicted_{}.npy'.format(index))
    # imgs_pre = np.load('./results/predicted_pre_{}.npy'.format(index))

    plt.figure()
    for i_image in range(5):
        img = test.item().get('input')[i_image]
        img = (1 - img)
        # img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, 1)
        plt.imshow(img)

        img = test.item().get('condition')[i_image]
        img = (1 - img)
        # img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, 2)
        plt.imshow(img)

        img = test.item().get('target')[i_image]
        img = (1 - img)
        # img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, 3)
        plt.imshow(img)

        for i in range(5):
            img = imgs[i, i_image]
            img = (1 - img)
            # img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            plt.subplot(2, 5, i + 6)
            plt.imshow(img)
        plt.show()
