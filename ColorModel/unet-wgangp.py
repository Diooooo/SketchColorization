import sys

sys.path.append('..')
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
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
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

    def train(self, dataset_path, list_files, epochs, mode_save_path, learning_rate=0.01, l1_rate=0.001,
              penalty_rate=10,
              save_index=1, restore_index=1):
        tf.reset_default_graph()
        input = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 1), name='input')
        condition = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                   name='condition')
        target = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3), name='output')
        is_training = tf.placeholder(tf.bool, name='is_training')
        random_e = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='random_e')

        g_logits = self.generator(input, condition, is_training)
        tf.add_to_collection('g_logits', g_logits)

        d_logits_real = self.discriminator(target, condition, reuse=False)
        d_logits_fake = self.discriminator(g_logits, condition, reuse=True)
        tf.add_to_collection('d_logits_real', d_logits_real)
        tf.add_to_collection('d_logits_fake', d_logits_fake)

        eps = 1e-16
        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g_logits - target), axis=[1, 2, 3]))
        g_loss = -tf.reduce_mean(d_logits_fake) + l1_rate * l1_loss

        x_hat = random_e * target + (1 - random_e) * g_logits
        d_logits_xhat = self.discriminator(x_hat, condition, reuse=True)
        grads = tf.gradients(d_logits_xhat, [x_hat])[0]
        penalty = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + eps) - 1))
        d_loss = tf.reduce_mean(d_logits_fake - d_logits_real) + penalty_rate * penalty
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

        data_parser = DataParserV2(dataset_path, self.image_shape, list_files=list_files, batch_size=self.batch_size)
        # data_parser_test = DataParserV2(dataset_path, self.image_shape, list_files=['../dataset/image_list_8.txt'],
        #                                 batch_size=20)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        losses = []
        losses_epoch = []

        # test_input = data_parser_test.get_batch_sketch()
        # test_target = data_parser_test.get_batch_raw()
        # test_condition = data_parser_test.get_batch_condition_add()
        # test = {'input': test_input, 'target': test_target, 'condition': test_condition}
        # np.save('./results/test_data_{}.npy'.format(save_index), test)

        test = np.load('./results/test_data_{}.npy'.format(restore_index)).item()
        outputs = []

        model_dir = os.path.join(mode_save_path, 'unet{}'.format(save_index))
        try:
            os.makedirs(model_dir)
        except FileExistsError as e:
            print(e)

        var_list = tf.trainable_variables('generator')
        # bn_var = [var for var in tf.global_variables('generator') if 'moving_mean' in var.name]
        # bn_var += [var for var in tf.global_variables('generator') if 'moving_variance' in var.name]
        # var_list += bn_var
        saver = tf.train.Saver(var_list)
        saver_model = tf.train.Saver(max_to_keep=100)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # init_var = [var for var in tf.global_variables() if 'generator' not in var.name]
        # for var in tf.global_variables():
        #     print(var)
        with tf.Session(config=config) as sess:
            # writer = tf.summary.FileWriter('./logs/train', sess.graph)
            # writer_val = tf.summary.FileWriter('./logs/val')

            # merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './checkpoints/unet_simple27/model-100.ckpt')

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
                        e = np.random.uniform(0, 1, [batch_target.shape[0], 1, 1, 1])
                        _, loss_d = sess.run([d_optimizer, d_loss],
                                             feed_dict={input: batch_input,
                                                        condition: batch_condition,
                                                        target: batch_target,
                                                        random_e: e,
                                                        is_training: True},
                                             options=run_options)

                    for _ in range(i_g):
                        _, loss_g, loss_l1 = sess.run([g_optimizer, g_loss, l1_loss],
                                                      feed_dict={input: batch_input,
                                                                 condition: batch_condition,
                                                                 target: batch_target,
                                                                 is_training: True},
                                                      options=run_options)
                    e = np.random.uniform(0, 1, [20, 1, 1, 1])
                    loss_d_val, loss_g_val, loss_l1_val = sess.run([d_loss, g_loss, l1_loss],
                                                                   feed_dict={input: test['input'],
                                                                              condition: test['condition'],
                                                                              target: test['target'],
                                                                              random_e: e,
                                                                              is_training: False})

                    losses.append([loss_d, loss_g, loss_d_val, loss_g_val, loss_l1])
                    losses_epoch.append([loss_d, loss_g, loss_d_val, loss_g_val, loss_l1_val])
                    if i % 10 == 0:
                        print(
                            'Epoch: {}, Iteration: {}/{}, g_loss: {}, d_loss: {}, l1:loss: {},'
                            ' g_loss_val: {}, d_loss_val: {}, l1_loss_val: {}'.format(
                                epoch + 1, i + 1, data_parser.iteration, loss_g, loss_d, loss_l1, loss_g_val,
                                loss_d_val, loss_l1_val))
                print('*' * 10,
                      'Epoch {}/{} ...'.format(epoch + 1, epochs),
                      'g_loss: {:.4f} ...'.format(loss_g),
                      'd_loss: {:.4f} ...'.format(loss_d),
                      'g_loss_val: {:.4f} ...'.format(loss_g_val),
                      'd_loss_val: {:.4f} ...'.format(loss_d_val),
                      'l1_loss: {:.4f} ...'.format(loss_l1),
                      'l1_loss_val: {:.4f} ...'.format(loss_l1_val),
                      '*' * 10)
                if (epoch + 1) % 10 == 0:
                    print('*' * 10, 'save results', '*' * 10)
                    output = sess.run(g_logits, feed_dict={input: test['input'], condition: test['condition'],
                                                           target: test['target'], is_training: False})
                    np.save('./results/predicted_pre_{}-{}.npy'.format(save_index, epoch + 1), np.asarray(output))
                    np.save('./logs/train/losses_{}-{}.npy'.format(save_index, epoch + 1), losses_epoch)
                    losses_epoch = []
                print('*' * 10, 'save model', '*' * 10)
                saver_model.save(sess, os.path.join(model_dir, 'checkpoint-{}.ckpt'.format(epoch + 1)))
            np.save('./results/predicted_{}.npy'.format(save_index), np.asarray(outputs))
        return losses


if __name__ == '__main__':
    dataset_path = '/media/bilin/MyPassport/data/dataset'
    resize_shape = (256, 256)
    list_files = ['../dataset/image_list_1.txt']
    batch_size = 10
    index = 31
    restore_index = 27
    l1_rate = 0.001
    penalty_rate = 10
    mode_save_path = '/media/bilin/MyPassport/data/checkpoints'

    model = Unet(resize_shape, batch_size=batch_size)

    losses = model.train(dataset_path, list_files, 100, mode_save_path, learning_rate=0.0001,
                         save_index=index, restore_index=restore_index, l1_rate=l1_rate, penalty_rate=penalty_rate)
    losses = np.asarray(losses)
    np.save('./logs/train/losses_{}.npy'.format(index), losses)
    # np.save('./logs/train/losses_pre_{}.npy'.format(index), np.asarray(losses_pretrain))

    # losses = np.load('./logs/train/losses_{}.npy'.format(index))
    # # losses_pretrain = np.load('./logs/train/losses_pre_{}.npy'.format(index))
    # # losses_pretrain = np.asarray(losses_pretrain)
    # plt.plot(losses[:, 0], label='train')
    # plt.plot(losses[:, 2], label='val')
    # plt.legend(loc='upper right')
    # plt.title('d loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # plt.plot(losses[:, 1], label='train')
    # plt.plot(losses[:, 3], label='val')
    # plt.legend(loc='upper right')
    # plt.title('g loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    # plt.plot(losses_pretrain)
    # plt.title('l1 loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    # test = np.load('./results/test_data_{}.npy'.format(restore_index))
    # # imgs = np.load('./results/predicted_{}.npy'.format(index))
    # # imgs_pre = np.load('./results/predicted_pre_{}.npy'.format(index))
    # unet_pre = np.load('./results/predicted_pre_{}-100.npy'.format(restore_index))
    # res_dir = './'
    # plt.figure()
    # for i_image in range(2,3):
    #     img = test.item().get('input')[i_image]
    #     img = (1 - img)
    #     cv2.imwrite(os.path.join(res_dir, 'input.jpg'), np.round(img*255))
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(4, 5, 1)
    #     plt.imshow(img)
    #
    #     img = test.item().get('condition')[i_image]
    #     img = (1 - img)
    #     cv2.imwrite(os.path.join(res_dir, 'condition.jpg'), np.round(img * 255))
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(4, 5, 2)
    #     plt.imshow(img)
    #
    #     img = test.item().get('target')[i_image]
    #     img = (1 - img)
    #     cv2.imwrite(os.path.join(res_dir, 'target.jpg'), np.round(img * 255))
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(4, 5, 3)
    #     plt.imshow(img)
    #
    #     img = unet_pre[i_image]
    #     img = (1 - img)
    #     cv2.imwrite(os.path.join(res_dir, 'pretrained.jpg'), np.round(img * 255))
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(4, 5, 4)
    #     plt.imshow(img)
    #
    #     for i in range(10):
    #         imgs = np.load('./results/predicted_pre_{}-{}.npy'.format(index, (i + 1) * 10))
    #         img = imgs[i_image]
    #         img = (1 - img)
    #         cv2.imwrite(os.path.join(res_dir, 'predicted-{}.jpg'.format((i + 1) * 10)), np.round(img * 255))
    #         # img = img.astype(np.uint8)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         # print(img.shape)
    #         plt.subplot(4, 5, i + 6)
    #         plt.imshow(img)
    #     plt.show()

    # sess = tf.Session()
    # new_saver = tf.train.import_meta_graph(os.path.join(mode_save_path, 'unet31/checkpoint-1.ckpt.meta'))
    # new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(mode_save_path, 'unet31')))
    #
    # data_parser = DataParserV2(dataset_path, resize_shape, list_files=list_files, batch_size=batch_size,
    #                            max_length=10)
    # graph = tf.get_default_graph()
    # inputs = graph.get_tensor_by_name('input:0')
    # condition = graph.get_tensor_by_name('condition:0')
    # target = graph.get_tensor_by_name('output:0')
    # is_train = graph.get_tensor_by_name('is_training:0')
    # random_e = graph.get_tensor_by_name('random_e:0')
    #
    # # logits = graph.get_tensor_by_name('generator/logits/Sigmoid:0')
    # logits = graph.get_collection('g_logits')
    #
    # e = np.random.uniform(0, 1, [10, 1, 1, 1])
    #
    # generated = sess.run(logits, feed_dict={inputs: data_parser.get_batch_sketch(),
    #                                         condition: data_parser.get_batch_color_hint(),
    #                                         target: data_parser.get_batch_raw(),
    #                                         random_e: e,
    #                                         is_train: False})
    # sess.close()
