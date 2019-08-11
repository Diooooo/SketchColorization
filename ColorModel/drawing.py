import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os


def draw_losses(data, labels, title, x_label, y_label, loc='upper right', save_png=None, res_dir='./'):
    """
    draw losses
    :param data: shape: (kinds of losses, losses data)
    :param labels: label for each kind of loss data
    :param title: title
    :param x_label: x lable
    :param y_label: y label
    :param loc: parameter of legend
    :param save_png: name of saved result, if None, the result would be dropped
    :param res_dir: directory to store the result figure
    :return: None
    """
    if data.shape[0] != len(labels):
        raise ValueError('labels and data not match')
    for i in range(data.shape[0]):
        plt.plot(data[i], label=labels[i])
    plt.legend(loc=loc)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_png and res_dir:
        plt.savefig(os.path.join(res_dir, save_png))
    plt.show()


def draw_generated(data, row, column, titles=None, save_png=None, res_dir='./'):
    """
    draw generated images
    :param data: shape: (num, height, width, channels)
    :param row: rows of subplots
    :param column: columns of subplots
    :param titles: title for each sub figure
    :param save_png: name of saved result, if None, the result would be dropped
    :param res_dir: directory to store the result figure
    :return: None
    """
    if titles:
        if data.shape[0] != len(titles):
            raise ValueError('titles and data not match')
    num_fig = int(min(data.shape[0], row * column))
    f, axarr = plt.subplots(row, column)
    for i in range(num_fig):
        if data.shape[-1] == 1:  # gray image
            fig = data[i].reshape(data.shape[1], data.shape[2])
            cmap = 'gray'
        else:
            fig = data[i]
            cmap = None
        axarr[int(i / column), i % column].imshow(fig, cmap=cmap)
        axarr[int(i / column), i % column].set_title(titles[i])
        axarr[int(i / column), i % column].axis('off')
    if save_png and res_dir:
        plt.savefig(os.path.join(res_dir, save_png))
    plt.show()


if __name__ == '__main__':
    pass
