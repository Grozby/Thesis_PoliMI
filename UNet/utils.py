import os
import time
from contextlib import contextmanager

import imageio
import matplotlib.pyplot as plt
import numpy as np


class Parser:
    """
    Utility parser used to obtain the data from the configuration file.
    """
    type_of_parsers = ["#IntegerValues", "#FloatValues", "#StringValues", "#BooleanValues", "#ListOfStringsValues"]

    def __init__(self, parser_type):
        super().__init__()
        self.serializer = self._get_parser(parser_type)

    def _get_parser(self, parser_type):
        if parser_type == "#IntegerValues":
            return self._parse_string_to_int
        if parser_type == "#FloatValues":
            return self._parse_string_to_float
        if parser_type == "#StringValues":
            return self._parse_string_to_string
        if parser_type == "#ListOfStringsValues":
            return self._parse_string_to_list_of_strings
        if parser_type == "#BooleanValues":
            return self._parse_string_to_boolean

        raise ValueError(parser_type)

    @staticmethod
    def _parse_string_to_int(string):
        return int(string)

    @staticmethod
    def _parse_string_to_float(string):
        return float(string)

    @staticmethod
    def _parse_string_to_list_of_strings(string):
        return string.split(",")

    @staticmethod
    def _parse_string_to_string(string):
        return string

    @staticmethod
    def _parse_string_to_boolean(string):
        return string in ["True", "true", "1"]

    def parse_string(self, string):
        return self.serializer(string)


def save_result(save_path, npyfile):
    # First we remove all the previous images
    file_list = [f for f in os.listdir(save_path) if f.endswith(".png")]
    for f in file_list:
        os.remove(os.path.join(save_path, f))

    for i, item in enumerate(npyfile):
        # We do this as our images are in grayscale!
        img = np.ceil(item[:, :, 0] * 255)
        imageio.imwrite(os.path.join(save_path, "{0}.png".format(i)), img.astype('uint8'))


@contextmanager
def measure_time(title):
    t1 = time.clock()
    yield
    t2 = time.clock()
    print('%s: %0.2f seconds elapsed' % (title, t2 - t1))


def plot_images(org_imgs,
                mask_imgs,
                pred_imgs,
                nm_img_to_plot=10,
                figsize=4,
                alpha=0.5,
                show=False,
                save=True
                ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    pred_imgs = reshape_arr(pred_imgs)
    cols = 4

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    axes[0, 2].set_title("prediction", fontsize=15)
    axes[0, 3].set_title("overlay", fontsize=15)

    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
        axes[m, 2].set_axis_off()
        axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)),
                          cmap=get_cmap(pred_imgs), alpha=alpha)
        axes[m, 3].set_axis_off()

        im_id += 1

    if show:
        plt.show()

    if save:
        plt.savefig("plotterino.png")


def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_height = mask.shape[0]
    img_width = mask.shape[1]
    c1 = mask.reshape(img_height, img_width)
    c2 = np.zeros((img_height, img_width))
    c3 = np.zeros((img_height, img_width))
    c4 = mask.reshape(img_height, img_width)
    return np.stack((c1, c2, c3, c4), axis=-1)


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask
