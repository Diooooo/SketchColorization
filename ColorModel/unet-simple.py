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


# base line
class UnetSimple:
    def __init__(self, image_shape, batch_size):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.params = []

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

    def train(self, dataset_path, list_files, epochs, learning_rate=0.01, save_index=1, l1_rate=0.1, l2_rate=1):
        tf.reset_default_graph()
        input = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 1), name='input')
        condition = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                   name='condition')
        target = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3), name='output')
        is_training = tf.placeholder(tf.bool, name='is_training')

        g_logits = self.generator(input, condition, is_training)

        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g_logits - target), axis=[1, 2, 3]))

        feature_fake = get_feature_map(g_logits)
        feature_real = get_feature_map(target, reuse=True)
        l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(feature_fake - feature_real), axis=1))

        g_loss = l1_rate * l1_loss + l2_rate * l2_loss

        var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_pretrain = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=var_g)

        data_parser = DataParserV2(dataset_path, self.image_shape, list_files=list_files, batch_size=self.batch_size)
        data_parser_test = DataParserV2(dataset_path, self.image_shape, list_files=['../dataset/image_list_8.txt'],
                                        batch_size=20)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        losses = []

        test_input = data_parser_test.get_batch_sketch()
        test_condition = data_parser_test.get_batch_color_hint()
        test_target = data_parser_test.get_batch_raw()

        test = {'input': test_input, 'condition': test_condition, 'target': test_target}
        np.save('./results/test_data_{}.npy'.format(save_index), test)
        outputs = []
        try:
            os.makedirs('./checkpoints/unet_simple{}'.format(save_index))
        except FileExistsError as e:
            print(e)
        var_list = tf.trainable_variables('generator')
        # bn_var = [var for var in tf.global_variables('generator') if 'moving_mean' in var.name]
        # bn_var += [var for var in tf.global_variables('generator') if 'moving_variance' in var.name]
        # var_list += bn_var
        saver = tf.train.Saver(var_list)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        with tf.Session(config=config) as sess:
            # writer = tf.summary.FileWriter('./logs/train_vgg', sess.graph)
            # writer_val = tf.summary.FileWriter('./logs/val')

            # merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())

            # pre-train generator
            for g_epoch in range(epochs):
                for i in range(data_parser.iteration):
                    batch_input = data_parser.get_batch_sketch()
                    batch_condition = data_parser.get_batch_color_hint()
                    batch_target = data_parser.get_batch_raw()
                    data_parser.update_iterator()

                    _, loss_g, loss_l1, loss_l2 = sess.run([g_pretrain, g_loss, l1_loss, l2_loss],
                                                           feed_dict={input: batch_input,
                                                                      target: batch_target,
                                                                      condition: batch_condition,
                                                                      is_training: True},
                                                           options=run_options)
                    losses.append([loss_g, loss_l1, loss_l2])
                    if i % 10 == 0:
                        print(
                            'Epoch: {}, Iteration: {}/{}, total_loss: {}, l1_loss: {}, l2_loss: {}'.format(
                                g_epoch + 1, i + 1, data_parser.iteration, loss_g, loss_l1, loss_l2))
                print('*' * 10,
                      'Epoch {}/{} ...'.format(g_epoch + 1, epochs),
                      'total_loss: {:.4f} ...'.format(loss_g),
                      'l1_loss: {:.4f} ...'.format(loss_l1),
                      'l2_loss: {:.4f} ...'.format(loss_l2),
                      '*' * 10)
                if (g_epoch + 1) % 100 == 0:
                    print('*' * 10, 'save results', '*' * 10)
                    output_pre = sess.run(g_logits, feed_dict={input: test_input,
                                                               target: test_target,
                                                               condition: test_condition,
                                                               is_training: False})
                    outputs.append(output_pre)

            for _ in range(10):
                print('*' * 10)

            saver.save(sess, './checkpoints/unet_simple{}/model.ckpt'.format(save_index))
            np.save('./results/predicted_pre_{}.npy'.format(save_index), np.asarray(outputs))
        return losses


if __name__ == '__main__':
    dataset_path = '/media/bilin/MyPassport/data/dataset'
    resize_shape = (128, 128)
    list_files = ['../dataset/image_list_1.txt']
    batch_size = 20
    l1_rate = 0.05
    l2_rate = 1
    index = 27

    # losses = np.load('./logs/train/losses_{}.npy'.format(index))

    model = UnetSimple(resize_shape, batch_size=batch_size)

    losses = model.train(dataset_path, list_files, 100, learning_rate=0.0001,
                         save_index=index, l1_rate=l1_rate, l2_rate=l2_rate)
    np.save('./logs/train/losses_{}.npy'.format(index), np.asarray(losses))
    # losses = np.asarray(losses)
    #
    # plt.plot(losses[:, 0])
    # plt.title('total loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # plt.plot(losses[:, 1])
    # plt.title('l1 loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # plt.plot(losses[:, 2])
    # plt.title('l2 loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # imgs_pre = np.load('./results/predicted_pre_{}.npy'.format(index))
    # test = np.load('./results/test_data_{}.npy'.format(index))
    #
    # plt.figure()
    # for i_image in range(5):
    #     img = test.item().get('input')[i_image]
    #     img = (1 - img)
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(3, 5, 1)
    #     plt.imshow(img)
    #
    #     img = test.item().get('condition')[i_image]
    #     img = (1 - img)
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(3, 5, 2)
    #     plt.imshow(img)
    #
    #     img = test.item().get('target')[i_image]
    #     img = (1 - img)
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.subplot(3, 5, 3)
    #     plt.imshow(img)
    #
    #     for i in range(10):
    #         img = imgs_pre[i * 5 + 4, i_image]
    #         img = (1 - img)
    #         # img = img.astype(np.uint8)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         # print(img.shape)
    #         plt.subplot(3, 5, i + 6)
    #         plt.imshow(img)
    #     plt.show()

    # sess = tf.Session()
    # new_saver = tf.train.import_meta_graph('./checkpoints/unet_simple27/model.ckpt.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/unet_simple27'))
    #
    # data_parser = DataParserV2(dataset_path, resize_shape, list_files=list_files, batch_size=batch_size,
    #                            max_length=10)
    # graph = tf.get_default_graph()
    # inputs = graph.get_tensor_by_name('input:0')
    # condition = graph.get_tensor_by_name('condition:0')
    # is_train = graph.get_tensor_by_name('is_training:0')
    #
    # logits = graph.get_tensor_by_name('generator/logits/Sigmoid:0')

    # generated = sess.run(logits, feed_dict={inputs: data_parser.get_batch_sketch(),
    #                                         condition: data_parser.get_batch_color_hint(),
    #                                         is_train: False})
    # sess.close()
    #
    # plt.figure()
    # for i in range(10):
    #     img = generated[i]
    #     img = (1 - img)
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # print(img.shape)
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(img)
    # plt.show()
    #
    # targets = data_parser.get_batch_raw()
    # plt.figure()
    # for i in range(10):
    #     img = targets[i]
    #     img = (1 - img)
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # print(img.shape)
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(img)
    # plt.show()

    # from tensorflow.python import pywrap_tensorflow
    # reader = pywrap_tensorflow.NewCheckpointReader('./checkpoints/unet_simple27/model.ckpt')
    # var_map = reader.get_variable_to_shape_map()
    # for key in var_map:
    #     print(key)
    #     print(reader.get_tensor(key).shape)
