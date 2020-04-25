import math
import random
from operator import itemgetter

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.batches import UnnormalizedBatch
from tensorflow import keras

import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb

import albumentations as A


def convert_rgba(image, shape):
    if image.shape[2] == 3:
        return image

    rgb = image[..., :3]
    brightness = image[..., 3, np.newaxis] / 255

    if shape[2] == 1:
        grayscale = np.mean(rgb, axis=2, keepdims=True)
        return np.array(grayscale * brightness, dtype=np.uint8)
    elif shape[2] == 3:
        return np.array(rgb * brightness, dtype=np.uint8)
    else:
        raise Exception("Invalid number of channels!")


class DataGenerator(keras.utils.Sequence):

    def __init__(self, images, labels,
                 batch_size=16,
                 image_shape=(256, 512, 1),
                 do_shuffle_at_epoch_end=True,
                 length=None,
                 do_augment=True):
        self.number_batches_augmentation = 4
        self.labels = labels  # array of labels
        self.images = images  # array of image paths
        self.input_shape = image_shape  # image dimensions
        self.label_shape = (image_shape[0], image_shape[1], 1)
        self.length = length
        self.batch_size = batch_size  # batch size
        self.shuffle = do_shuffle_at_epoch_end  # shuffle bool
        self.augment = do_augment  # augment data bool
        self.augmenting_pipeline = iaa.Sequential([
            iaa.PadToFixedSize(pad_mode="symmetric", width=1024, height=0),
            iaa.Affine(shear=(-20, 20)),
            iaa.CropToFixedSize(width=512, height=256, position="center"),
            iaa.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)},
                       translate_percent={"x": (-0.4, 0.4)},
                       mode="symmetric",
                       cval=[0, 0, 0, 0]),
            iaa.PerspectiveTransform(scale=(0, 0.10)),
            iaa.CropToFixedSize(width=512, height=256, position="center-top"),
            # iaa.PiecewiseAffine(scale=(0.001, 0.01), nb_rows=4, nb_cols=8),
            iaa.Lambda(func_images=self.add_background,
                       func_segmentation_maps=self.convert_segmentations),

        ])
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        if self.length is None:
            return int(np.ceil(len(self.images) / self.batch_size))
        elif self.length == 'len':
            return len(self.images)
        else:
            raise ValueError("Value {} not valid for parameter step_per_epoch")

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if self.length is not None:
            index = index % self.batch_size

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        labels = np.array([self.labels[k] for k in indexes])
        images = np.array([self.images[k] for k in indexes])

        if self.augment:
            images, labels = self.augmentation(images, labels)

        images = np.array(images, dtype=np.float32) / 255
        labels = np.array(labels, dtype=np.float32) / 255
        return images, labels

    @staticmethod
    def generate_background_noise(image_shape):
        random_state = int(np.random.rand() * 100000)

        background_noise = np.zeros(image_shape, dtype=np.uint8)
        background_noise[:] = [0, 0, 0, 255]

        noise_augmentation = iaa.Sequential([
            iaa.Salt(p=0.041, random_state=random_state),
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.MotionBlur(angle=90, k=8),
            iaa.GaussianBlur(sigma=0.6)
        ])

        return noise_augmentation(image=background_noise)

    def add_background(self, images, random_state=None, parents=None, hooks=None):
        processed_images = []
        for img in images:
            # Adding background noise
            noise = self.generate_background_noise(img.shape)
            non_transparent_mask = img[..., 3] != 0
            non_transparent_mask = np.repeat(non_transparent_mask, 4).reshape((non_transparent_mask.shape[0],
                                                                               non_transparent_mask.shape[1],
                                                                               4))
            noise[non_transparent_mask] = img[non_transparent_mask]

            processed_images.append(convert_rgba(noise, self.input_shape))
        return processed_images

    def convert_segmentations(self, segmentation_images, random_state=None, parents=None, hooks=None):
        for segmentation in segmentation_images:
            segmentation.arr = convert_rgba(segmentation.arr, self.label_shape)
        return segmentation_images

    def augmentation(self, images, labels):
        batches = []
        batch_size = math.ceil(self.batch_size / self.number_batches_augmentation)

        for i in range(self.number_batches_augmentation - 1):
            batches.append(UnnormalizedBatch(
                images=images[i * batch_size: (i + 1) * batch_size],
                segmentation_maps=labels[i * batch_size: (i + 1) * batch_size]
            ))

        batches.append(UnnormalizedBatch(
            images=images[(self.number_batches_augmentation - 1) * batch_size: self.batch_size],
            segmentation_maps=labels[(self.number_batches_augmentation - 1) * batch_size: self.batch_size]
        ))

        # time_start = time.time()
        batches_aug = list(self.augmenting_pipeline.augment_batches(batches, background=True))
        # time_end = time.time()

        # print("Augmentation done in %.2fs" % (time_end - time_start,))
        return [image for batch in batches_aug for image in batch.images_aug], \
               [label for batch in batches_aug for label in batch.segmentation_maps_aug]


if __name__ == "__main__":
    print("Nope")


class DataGeneratorAlbumentations(DataGenerator):
    def __init__(self, images, labels, batch_size=16,
                 image_shape=(256, 512, 1),
                 do_shuffle_at_epoch_end=True,
                 length=None,
                 do_augment=True):
        super().__init__(images, labels, batch_size,
                         image_shape,
                         do_shuffle_at_epoch_end,
                         length,
                         do_augment)
        self.augmenting_pipeline = A.Compose([
            A.HorizontalFlip(),
            A.IAAAffine(translate_percent={"x": (-1, 1)},
                        mode="reflect",
                        p=1),
            A.PadIfNeeded(min_width=int(self.input_shape[1] * 2),
                          min_height=self.input_shape[0]),
            A.GridDistortion(p=0.8, distort_limit=0.5),
            A.ElasticTransform(p=0.5, alpha=10, sigma=100 * 0.03, alpha_affine=0),

            A.CenterCrop(width=self.input_shape[1], height=self.input_shape[0]),
            A.IAAPerspective(scale=(0, 0.10), p=1),
            A.ShiftScaleRotate(shift_limit=0,
                               scale_limit=(.0, 0.4),
                               rotate_limit=0,
                               p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.Lambda(
                image=self.convert_image,
                mask=self.convert_segmentations,
            ),
        ])

    def __getitem__(self, index):
        if self.length is not None:
            index = index % int(np.ceil(len(self.images) / self.batch_size))

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        labels = np.array([self.labels[k] for k in indexes])
        images = np.array([self.images[k] for k in indexes])

        if self.augment:
            augmentation = [
                itemgetter("image", "mask")(self.augmenting_pipeline(image=images[i], mask=labels[i]))
                for i in range(len(indexes))
            ]
            images, labels = map(list, zip(*augmentation))

        images = np.array(images, dtype=np.float32) / 255
        labels = np.array(labels, dtype=np.float32) / 255
        return images, labels

    def convert_image(self, image, **kwargs):
        return convert_rgba(image, self.input_shape)

    def convert_segmentations(self, label, **kwargs):
        non_black_pixels_mask = ~np.all(label[..., :3] == [0, 0, 0], axis=-1)
        label[non_black_pixels_mask] = [255, 255, 255, 255]

        return convert_rgba(label, self.label_shape)
