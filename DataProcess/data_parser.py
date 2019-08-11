import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class DataParser:
    """
    Simplest data parser, can only generate batch from one numpy array
    """
    def __init__(self, inputs, batch_size):
        self.inputs = inputs
        self.iterator = 0
        self.batch_size = batch_size
        self.m = inputs.shape[0]
        self.iteration = int(np.ceil(self.m / self.batch_size))
        self.indices = np.random.permutation(self.m)

    def get_batch(self):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            self.iterator = 0
            self.indices = np.random.permutation(self.m)
        return self.inputs[batch_indices]


class DataParserV2:
    """
    generate batch data from given files, which request I/O operation every time the get_batch_{} function is called
    """
    def __init__(self, dataset_path, resize_shape, list_files, batch_size, max_length=None):
        """

        :param dataset_path: should contains ['/raw_image', 'sketch', 'color_hint', 'color_hint_with_whiteout', 'color_block']
        :param resize_shape: resize shape
        :param list_files: files which contains the names of images
        :param batch_size: batch size
        :param max_length: max number of images to be read from list_files
        """
        self.dataset_path = dataset_path
        self.resize_shape = resize_shape
        self.list_files = list_files
        self.batch_size = batch_size

        self.raw_image_path = os.path.join(self.dataset_path, 'raw_image')
        self.sketch_path = os.path.join(self.dataset_path, 'sketch')
        self.color_hint_path = os.path.join(self.dataset_path, 'color_hint')
        self.color_hint_whiteout_path = os.path.join(self.dataset_path, 'color_hint_with_whiteout')
        self.color_block_path = os.path.join(self.dataset_path, '../dataset/color_block')
        self.images_name = []
        for lf in self.list_files:
            with open(lf, 'r') as f:
                name = f.readline()
                while name:
                    self.images_name.append(name.strip('\n'))
                    name = f.readline()
        self.m = len(self.images_name)
        self.iteration = int(np.ceil(self.m / self.batch_size))
        self.iterator = 0
        self.indices = np.random.permutation(self.m)
        if max_length and max_length < self.m:
            self.indices = self.indices[:max_length]
            self.m = max_length
            self.iteration = int(np.ceil(self.m / self.batch_size))

    def _get_indices(self, update=False):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            if update:
                self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            if update:
                self.iterator = 0
                np.random.shuffle(self.indices)
        return batch_indices

    def update_iterator(self):
        """
        need to be called anytime you need to get next batch
        """
        self._get_indices(True)

    def get_batch_raw(self):
        """
        get batch of original images
        """
        indices = self._get_indices()
        raws = []
        for id in indices:
            raw_name = os.path.join(self.raw_image_path, self.images_name[id])
            raw_img = cv2.imread(raw_name).astype(np.float32)
            raw_img = cv2.resize(raw_img, self.resize_shape)
            raws.append((1 - raw_img / 255))
        return np.asarray(raws)

    def get_batch_raw_gray(self):
        """
        get batch of gray images
        """
        indices = self._get_indices()
        raws_gray = []
        for id in indices:
            raw_name = os.path.join(self.raw_image_path, self.images_name[id])
            raw_img = cv2.imread(raw_name).astype(np.float32)
            raw_img = cv2.resize(raw_img, self.resize_shape)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            raw_img = np.expand_dims(raw_img, axis=2)
            raws_gray.append((1 - raw_img / 255))
        return np.asarray(raws_gray)

    def get_batch_sketch(self):
        """
        get batch of sketches
        """
        indices = self._get_indices()
        sketches = []
        for id in indices:
            sketch_name = os.path.join(self.sketch_path, self.images_name[id].split('.')[0] + '_sketch.jpg')
            sketch = cv2.imread(sketch_name, 0).astype(np.float32)
            sketch = cv2.resize(sketch, self.resize_shape)
            noise = np.random.normal(loc=0, scale=0.1, size=1000)
            pos = np.random.permutation(self.resize_shape[0] * self.resize_shape[1])[:1000]
            noise_channel = np.zeros_like(sketch)
            for i, val in enumerate(pos):
                noise_channel[val // self.resize_shape[0]][val % self.resize_shape[1]] = noise[i]
            sketch = np.expand_dims(sketch, axis=2)
            noise_channel = np.expand_dims(noise_channel, axis=2)
            sketch = 1 - sketch / 255 + noise_channel
            # sketch = 1 - sketch / 255
            # sketch = np.concatenate((sketch, noise_channel), axis=2)
            sketches.append(sketch)
        return np.asarray(sketches)

    def get_batch_color_hint(self):
        """
        get batch of color hint which is just the gaussian-filtered images
        """
        indices = self._get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_hint_path, self.images_name[id].split('.')[0] + '_colorhint.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append((1 - img / 255))
        return np.asarray(res)

    def get_batch_color_hint_whiteout(self):
        """
        get batch of color hint which added whiteout blocks
        """
        indices = self._get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_hint_whiteout_path, self.images_name[id].split('.')[0] + '_whiteout.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append((1 - img / 255))
        return np.asarray(res)

    def get_batch_color_block(self):
        """
        get batch of color block, the background is white
        """
        indices = self._get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_block_path, self.images_name[id].split('.')[0] + '_colorblock.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append((1 - img / 255))
        return np.asarray(res)

    def get_batch_condition(self):
        """
        get batch of the concatenated color hint(whiteout) and color block
        """
        white_outs = self.get_batch_color_hint_whiteout()
        color_blocks = self.get_batch_color_block()
        return np.concatenate((white_outs, color_blocks), axis=3)

    def get_batch_condition_add(self):
        """
        get batch of condition which replaces the specific position with color block
        notice this function require more I/O operation
        """
        indices = self._get_indices()
        res = []
        for id in indices:
            name_block = os.path.join(self.color_block_path, self.images_name[id].split('.')[0] + '_colorblock.jpg')
            name_whiteout = os.path.join(self.color_hint_whiteout_path,
                                         self.images_name[id].split('.')[0] + '_whiteout.jpg')
            img = cv2.imread(name_block).astype(np.float32)
            img_block = cv2.resize(img, self.resize_shape)
            img_block = 1 - img_block / 255

            img = cv2.imread(name_whiteout).astype(np.float32)
            img_whiteout = cv2.resize(img, self.resize_shape)
            img_whiteout = 1 - img_whiteout / 255

            tmp = np.sum(img_block, axis=2)
            # print(len(tmp[tmp > 0]))
            # print(len(tmp[tmp == 0]))
            for i in range(img_block.shape[0]):
                for j in range(img_block.shape[1]):
                    if tmp[i, j] > 0.1:
                        img_whiteout[i, j, :] = img_block[i, j, :]
            res.append(img_whiteout)
        return np.asarray(res)


class DataParserV3:
    """
    get batch from numpy file, the parameters' setting is almost the same as DataParserV2
    this class currently can only load one set of data due to the memory capacity
    """
    def __init__(self, dataset_path, resize_shape, index, batch_size, max_length=None):
        """
        :param dataset_path: path to load numpy files
        :param resize_shape: resize shape
        :param index: index of dataset to be loaded
        :param batch_size: batch size
        :param max_length: same as V2
        """
        self.dataset_path = dataset_path
        self.resize_shape = resize_shape
        self.batch_size = batch_size

        self.raw_images = np.load(os.path.join(os.path.join(self.dataset_path, '{:03d}'.format(index)), 'raw.npy'))
        self.sketches = np.load(os.path.join(os.path.join(self.dataset_path, '{:03d}'.format(index)), 'sketch.npy'))
        # self.color_hints = np.load(
        #     os.path.join(os.path.join(self.dataset_path, '{:03d}'.format(index)), 'color_hint.npy'))
        self.color_hint_whiteouts = np.load(
            os.path.join(os.path.join(self.dataset_path, '{:03d}'.format(index)), 'whiteout.npy'))
        self.color_blocks = np.load(
            os.path.join(os.path.join(self.dataset_path, '{:03d}'.format(index)), 'color_block.npy'))

        self.m = len(self.raw_images.shape[0])
        self.iteration = int(np.ceil(self.m / self.batch_size))
        self.iterator = 0
        self.indices = np.random.permutation(self.m)
        if max_length and max_length < self.m:
            self.indices = self.indices[:max_length]
            self.m = max_length
            self.iteration = int(np.ceil(self.m / self.batch_size))

    def get_indices(self, update=False):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            if update:
                self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            if update:
                self.iterator = 0
                np.random.shuffle(self.indices)
        return batch_indices

    def update_iterator(self):
        self.get_indices(True)

    def get_batch_raw(self):
        indices = self.get_indices()
        raws = self.raw_images[indices].astype(np.float32)
        return 1 - raws / 255

    def get_batch_sketch(self):
        indices = self.get_indices()
        sketches = self.sketches[indices]
        noise_channel = np.zeros_like(sketches)
        for i in range(len(indices)):
            noise = np.random.normal(loc=0, scale=0.1, size=1000)
            pos = np.random.permutation(self.resize_shape[0] * self.resize_shape[1])[:1000]
            for j, val in enumerate(pos):
                noise_channel[i, val // self.resize_shape[0], val % self.resize_shape[1]] = noise[j]
        sketches = 1 - sketches / 255 + noise_channel
        return np.expand_dims(sketches, axis=3)

    # def get_batch_color_hint(self):
    #     indices = self.get_indices()
    #     res = self.color_hints[indices].astype(np.float32)
    #     return 1 - res / 255

    def get_batch_color_hint_whiteout(self):
        indices = self.get_indices()
        res = self.color_hint_whiteouts[indices].astype(np.float32)
        return 1 - res / 255

    def get_batch_color_block(self):
        indices = self.get_indices()
        res = self.color_blocks[indices].astype(np.float32)
        return 1 - res / 255

    def get_batch_condition(self):
        white_outs = self.get_batch_color_hint_whiteout()
        color_blocks = self.get_batch_color_block()
        return np.concatenate((white_outs, color_blocks), axis=3)

    def get_batch_condition_add(self):
        white_outs = self.get_batch_color_hint_whiteout()
        color_blocks = self.get_batch_color_block()
        white_outs[color_blocks > 0.1] = color_blocks[color_blocks > 0.1]
        return white_outs


if __name__ == "__main__":
    data_parser = DataParserV3('/media/bilin/MyPassport/data/dataset', (256, 256), 1, batch_size=1)
    condition = data_parser.get_batch_condition_add()
    raw = data_parser.get_batch_raw()
    raw = (1 - raw) * 255
    raw = cv2.cvtColor(raw[0], cv2.COLOR_BGR2RGB)

    whiteout = data_parser.get_batch_color_hint_whiteout()
    whiteout = (1 - whiteout) * 255
    whiteout = cv2.cvtColor(whiteout[0], cv2.COLOR_BGR2RGB)

    condition = data_parser.get_batch_condition_add()
    condition = (1 - condition) * 255
    condition = cv2.cvtColor(condition[0], cv2.COLOR_BGR2RGB)

    sketch = data_parser.get_batch_sketch()
    sketch = (1 - sketch) * 255
    print(sketch.shape)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(raw.astype(np.uint8))
    plt.subplot(2, 2, 2)
    plt.imshow(whiteout.astype(np.uint8))
    plt.subplot(2, 2, 3)
    plt.imshow(sketch[0].reshape(256, 256), cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(condition)
    plt.show()
