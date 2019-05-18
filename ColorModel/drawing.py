import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os


def draw_loss(losses):
    pass


if __name__ == '__main__':
    index = 27
    restore_index = 27
    losses = None
    res_dir = './results_img'
    # for i in range(7):
    #     if losses is None:
    #         losses = np.load('./logs/train/losses_{}-{}.npy'.format(index, (i + 1) * 10))
    #     else:
    #         losses = np.concatenate((losses, np.load('./logs/train/losses_{}-{}.npy'.format(index, (i + 1) * 10))),
    #                                 axis=0)
    losses = np.load('./logs/train/losses_{}.npy'.format(index))
    # plt.plot(losses[:, 0], label='train')
    # plt.plot(losses[:, 2], label='val', alpha=0.9)
    # plt.legend(loc='upper right')
    # plt.title('d loss--unet+wgangp')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.savefig(os.path.join(res_dir, 'dloss-wgangp.png'))
    # plt.show()
    #
    # plt.plot(losses[:, 1], label='train')
    # plt.plot(losses[:, 3], label='val', alpha=0.9)
    # plt.legend(loc='upper right')
    # plt.title('g loss--unet+wgangp')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.savefig(os.path.join(res_dir, 'gloss-wgangp.png'))
    # plt.show()

    plt.plot(losses[:, 1], label='train')
    plt.plot(losses[:, 4], label='val', alpha=0.9)
    plt.legend(loc='upper right')
    plt.title('l1 loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(res_dir, 'l1-loss-unet.png'))
    plt.show()

    plt.plot(losses[:, 2], label='train')
    plt.plot(losses[:, 5], label='val', alpha=0.9)
    plt.legend(loc='upper right')
    plt.title('l2 loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(res_dir, 'l2-loss-unet.png'))
    plt.show()

    # test = np.load('./results/test_data_{}.npy'.format(restore_index))
    # # imgs = np.load('./results/predicted_{}.npy'.format(index))
    # # imgs_pre = np.load('./results/predicted_pre_{}.npy'.format(index))
    # unet_pre = np.load('./results/predicted_pre_{}-100.npy'.format(restore_index))

    # imgs = np.load('./results/predicted_pre_{}-{}.npy'.format(index, 70))
    # for i_image in range(20):
    #     img = imgs[i_image]
    #     img = (1 - img)
    #     cv2.imwrite(os.path.join(res_dir, 'wgangp-{}.jpg'.format(i_image + 1)),
    #                 np.round(img * 255))
    #     # img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # print(img.shape)
    #     plt.show()
