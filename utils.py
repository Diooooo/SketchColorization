import cv2
import numpy as np
import os
import time
import shutil


def generate_img_file(img_dir, save_path, threshold=10):
    """
    clean the collected images and generate image files, one file contains at most 10k names
    :param img_dir: directory of collected images
    :param save_path: path to save imgae files
    :param threshold: explained inside the code
    :return: None
    """
    name_list = []
    img_names = os.listdir(img_dir)
    count = 0
    pre = time.time()
    broken_imgs = []
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            print('Passed [{}]--Broken'.format(name))
            broken_imgs.append(name)
            continue
        if min(img.shape[0], img.shape[1]) < 256:
            print('Passed [{}]--Small'.format(name))
            continue
        ''' data cleaning,  filtered non-colorful images via saturation'''
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s_w, s_h = s.shape[:2]
        s_sum = np.sum(s) / (s_w * s_h)
        if s_sum < threshold:
            print('Passed [{}]--Is a Sketch'.format(name))
            continue
        name_list.append(name)
        count += 1
        print('Image [{}] saved, already saved {:d}'.format(name, count))
        if count % 10000 == 0 and count > 0:
            with open(os.path.join(save_path, 'image_list_{:d}.txt'.format(int(count // 10000))), 'w') as f:
                for i in name_list:
                    f.write(i)
                    f.write('\n')
                # f.write(str(name_list))
                print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                      ' ({:d}--{:d})'.format(count - 10000, count), '*' * 20)
                used = time.time() - pre
                print('Using {:d}min {:d}s'.format(int(used / 60), int(used % 60)))
                pre = time.time()
            name_list = []
    if count % 10000 != 0:
        with open(os.path.join(save_path, 'image_list_{:d}.txt'.format(int(count // 10000))), 'w') as f:
            for i in name_list:
                f.write(i)
                f.write('\n')
            # f.write(str(name_list))
            print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                  ' ({:d}--{:d})'.format(count - count % 10000, count), '*' * 20)
    for broken in broken_imgs:
        try:
            os.remove(os.path.join(img_dir, broken))
            print('Delete {}'.format(broken))
        except NotImplementedError as e:
            print('Delete {} fail'.format(broken))


def move_files_by_namelist(img_dir, save_path, list_file):
    """
    move images to assigned directories
    """
    num = int(list_file.split('_')[2].split('.')[0])
    if not os.path.exists(os.path.join(save_path, '{:03d}'.format(num))):
        os.mkdir(os.path.join(save_path, '{:03d}'.format(num)))
    with open(list_file, 'r+') as f:
        names = f.readlines()
        f.seek(0, 0)
        for name in names:
            new_name = '{:03d}/'.format(num) + name
            shutil.move(os.path.join(img_dir, name.strip('\n')), os.path.join(save_path, '{:03d}'.format(num)))
            f.write(new_name)
            print('Move ', name, ' to ', new_name)


def images2npy(data_dir, list_file, resize_shape, save_dir):
    """
    currently, it can only deal with one list file, due to the memory capacity
    :param data_dir: path of dataset, should contains "raw_image", "sketch", "color_hint", "color_hint_with_whiteout"
     and "color_block"
    :param list_file: file contains name of images
    :param resize_shape: resize shape
    :param save_dir: save path
    :return: None
    """
    with open(list_file) as f:
        img_names = f.readlines()
    index = list_file.split('_')[-1].split('.')[0]
    raw_imgs = []
    sketches = []
    color_hints = []
    whiteouts = []
    color_blocks = []
    for i, name in enumerate(img_names):
        img_name = name.strip('\n')
        raw_image = cv2.imread(os.path.join(os.path.join(data_dir, 'raw_image'), img_name))
        raw_image = cv2.resize(raw_image, resize_shape)

        img_num = img_name.split('/')[-1].split('.')[0]  # 'str'

        sketch = cv2.imread(
            os.path.join(os.path.join(data_dir, 'sketch/{:03d}'.format(int(index))), img_num + '_sketch.jpg'))
        sketch = cv2.resize(sketch, resize_shape)

        color_hint = cv2.imread(
            os.path.join(os.path.join(data_dir, 'color_hint/{:03d}'.format(int(index))), img_num + '_colorhint.jpg'))
        color_hint = cv2.resize(color_hint, resize_shape)

        whiteout = cv2.imread(
            os.path.join(os.path.join(data_dir, 'color_hint_with_whiteout/{:03d}'.format(int(index))),
                         img_num + '_whiteout.jpg'))
        whiteout = cv2.resize(whiteout, resize_shape)

        color_block = cv2.imread(
            os.path.join(os.path.join(data_dir, 'color_block/{:03d}'.format(int(index))), img_num + '_colorblock.jpg'))
        color_block = cv2.resize(color_block, resize_shape)

        raw_imgs.append(raw_image.astype(np.uint8))
        sketches.append(sketch.astype(np.uint8))
        color_hints.append(color_hint.astype(np.uint8))
        whiteouts.append(whiteout.astype(np.uint8))
        color_blocks.append(color_block.astype(np.uint8))
        print('Saved [{}]'.format(img_name), '--no.{}'.format(i + 1))
    os.mkdir(os.path.join(save_dir, '{:03d}'.format(int(index))))
    np.save(os.path.join(os.path.join(save_dir, '{:03d}'.format(int(index))), 'raw.npy'), raw_imgs)
    np.save(os.path.join(os.path.join(save_dir, '{:03d}'.format(int(index))), 'sketch.npy'), sketches)
    np.save(os.path.join(os.path.join(save_dir, '{:03d}'.format(int(index))), 'color_hint.npy'), color_hints)
    np.save(os.path.join(os.path.join(save_dir, '{:03d}'.format(int(index))), 'whiteout.npy'), whiteouts)
    np.save(os.path.join(os.path.join(save_dir, '{:03d}'.format(int(index))), 'color_block.npy'), color_blocks)
    print('*' * 10 + 'Saved [{}]'.format(len(img_names)) + ' from ', list_file + '*' * 10)


if __name__ == "__main__":
    # generate_img_file('/media/bilin/MyPassport/zerochain', './dataset')
    # for i in range(1, 9, 1):
    #     list_file = './dataset/image_list_{:d}.txt'.format(i)
    #     move_files_by_namelist('/media/bilin/MyPassport/zerochain', '/media/bilin/MyPassport/zerochain', list_file)
    #     print('*' * 20, 'Success move images listed in ', list_file, '*' * 20)

    list_file = './dataset/image_list_1.txt'
    dataset_path = '/media/bilin/MyPassport/data/dataset'
    save_path = './dataset'
    resize_shape = (256, 256)
    images2npy(dataset_path, list_file, resize_shape, save_path)
