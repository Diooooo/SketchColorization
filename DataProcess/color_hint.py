import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter


# import tensorflow as tf


def median_pyramid(img, filter_size=2):
    """
    process of average pooling
    :param img: image, with shape(m, n)
    :param filter_size: filter size z
    :return: result image with shape (m/z, n/z)
    """
    m, n, channels = img.shape[0], img.shape[1], img.shape[2]
    channels = img.shape[2]
    res = np.zeros((m // filter_size, n // filter_size, channels))
    for k in range(channels):
        for i in range(m // filter_size):
            for j in range(n // filter_size):
                res[i, j, k] = np.median(
                    (img[i * filter_size:(i + 1) * filter_size, j * filter_size:(j + 1) * filter_size, k]))
    return res


def generate_color_map(img):
    """
    calculate color map of a single image, here we use one Gaussian filter with variance=35 and then resize to
    the original shape
    :param img: original image(single)
    :return: color map(as the initial color hit)
    """
    # blurred = median_pyramid(img, 4)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    # blurred = median_pyramid(blurred, 4)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    # blurred = median_pyramid(blurred, 4)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    # blurred = cv2.resize(blurred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    blurred = np.zeros_like(img)
    for i in range(3):
        blurred[:, :, i] = gaussian_filter(img[:, :, i], 35)
    return blurred.astype(np.uint8)


def generate_whiteout(img, block_shape, block_num):
    """
    add whiteout to the color map with fixed block shape and number
    :param img: color map(or raw image if you want)
    :param block_shape: fixed block shape
    :param block_num: fixed block number
    :return: image with whiteout
    """
    if min(block_shape) > min(img.shape[:2]) - 1:
        raise ValueError('Block too large')
    output = img.copy()
    for _ in range(block_num):
        m = random.randint(0, img.shape[0] - block_shape[0])
        n = random.randint(0, img.shape[1] - block_shape[1])
        output[m:m + block_shape[0], n:n + block_shape[1], :] = 255
    return output


def generate_color_block(origin_img, img, block_shape, block_num):
    """
    generate color blocks from original image and store them in a given image(it can be anything), with fixed block shape
    and number
    :param origin_img: original image
    :param img: image to store the color blocks
    :param block_shape: fixed block shape
    :param block_num: fixed block shape
    :return: image where store the color blocks
    """
    if not origin_img.shape == img.shape:
        raise ValueError('Original image and input image must have the same shape')
    if min(block_shape) > min(img.shape[:2]) - 1:
        raise ValueError('Shape of original image is ' + str(origin_img.shape) + ', Block too large')
    iter = 0
    output = img.copy()
    while iter < block_num:
        m = random.randint(0, img.shape[0] - block_shape[0])
        n = random.randint(0, img.shape[1] - block_shape[1])
        block = origin_img[m:m + block_shape[0], n:n + block_shape[1], :]
        var = np.sum(np.var(block, axis=(0, 1)))
        # if var:
        #     continue
        output[m:m + block_shape[0], n:n + block_shape[1], :] = np.average(block, axis=(0, 1)).reshape(1, 1, 3)
        iter += 1
    return output


def generate_color_block_random_normal(origin_img, img, block_shape_miu, block_num, block_shape_sigma=6):
    if not origin_img.shape == img.shape:
        raise ValueError('Original image and input image must have the same shape')
    if block_shape_miu + block_shape_sigma > min(img.shape[:2]) - 1:
        raise ValueError('Shape of original image is ' + str(origin_img.shape) + ', Block too large')

    iter = 0
    output = img.copy()
    while iter < block_num:
        block_shape = np.int(np.ceil(abs(np.random.normal(loc=block_shape_miu, scale=block_shape_sigma, size=1)[0])))
        # print(block_shape)
        m = random.randint(0, img.shape[0] - block_shape)
        n = random.randint(0, img.shape[1] - block_shape)
        block = origin_img[m:m + block_shape, n:n + block_shape, :]
        var = np.sum(np.var(block, axis=(0, 1)))
        print(block_shape, m, n)
        if var > 1000:
            continue
        output[m:m + block_shape, n:n + block_shape, :] = np.average(block, axis=(0, 1)).reshape(1, 1, 3)
        cv2.rectangle(output, (n, m), (n + block_shape, m + block_shape), (0, 0, 255), 1)
        iter += 1
    return output


if __name__ == "__main__":
    img_path = '../demo3.jpg'
    img = cv2.imread(img_path)
    # erosion = cv2.erode(img, np.ones((1, 1)), iterations=2)
    # dilation = cv2.dilate(img, np.ones((3, 3)), iterations=1)
    # edge = dilation - erosion
    # cv2.imshow("", generate_color_map(img))
    # cv2.waitKey()
    img = cv2.resize(img, (512, 512))
    blurred = generate_color_map(img)
    # blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    blocked = generate_color_block_random_normal(origin_img=img, img=img, block_num=5, block_shape_miu=5,
                                                 block_shape_sigma=6)
    cv2.imshow('demo3', blocked)
    cv2.waitKey()
    # blocked = cv2.cvtColor(blocked, cv2.COLOR_BGR2RGB)
    # plt.imshow(blocked)
    # plt.show()
