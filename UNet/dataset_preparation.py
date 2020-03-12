from __future__ import print_function

import itertools
import os

import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

dirname = os.path.dirname(__file__)


def prepare_dataset(**configuration):
    """
    Function that searches for all the images that have been labeled.
    """
    path_prediction_images = os.path.join(dirname, "dataset/Prediction images - full retina/")
    path_training_images = os.path.join(dirname, "dataset/Resized images/")

    # First, we obtain all the directories in our prediction images.
    directories = []
    for (_, directory_names, _) in os.walk(path_prediction_images):
        directories = directory_names
        break

    # Then, we iterate over these directories, and we obtain the corresponding
    # image-mask pair for our dataset.
    x, y, x_test, y_test = (np.empty((0, 256, 512, 1), dtype=np.uint8) for _ in range(4))

    for directory in directories:
        images = []
        masks = []
        for (_, _, filenames) in os.walk(path_prediction_images + directory):
            for filename in filenames:
                if ".png" in filename:
                    # First, we load the images and immediately normalize them.
                    image = np.array(Image.open(path_training_images + directory + '/' + filename),
                                     dtype=np.uint8)
                    image = image.reshape(image.shape[0], image.shape[1], 1)
                    mask = 255 - np.array(Image.open(path_prediction_images + directory + '/' + filename),
                                          dtype=np.uint8)
                    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

                    images.append(image)
                    masks.append(mask)

        if images:
            if "validation_cubes" in configuration and directory in configuration["validation_cubes"]:
                x_test = np.append(x_test, np.array(images), axis=0)
                del images
                y_test = np.append(y_test, np.array(masks), axis=0)
                del masks
            else:
                x = np.append(x, np.array(images), axis=0)
                del images
                y = np.append(y, np.array(masks), axis=0)
                del masks

            print("Read directory {}...".format(directory))

    print("Splitting the dataset...")
    # Creation of train and validation set
    x_train, x_val, y_train, y_val = \
        train_test_split(x, y,
                         test_size=configuration['test_and_validation_size'],
                         random_state=configuration['seed'])

    print("Deleting complete dataset...")
    del x
    del y

    if "validation_cubes" not in configuration:
        # From the validation test we obtain the test set.
        x_test, x_val, y_test, y_val = \
            train_test_split(x_val,
                             y_val,
                             test_size=configuration['val_size_of_test_and_size'],
                             random_state=configuration['seed'])

    x_train = np.array(x_train, dtype=np.float32) / 255
    y_train = np.array(y_train, dtype=np.float32) / 255
    x_val = np.array(x_val, dtype=np.float32) / 255
    y_val = np.array(y_val, dtype=np.float32) / 255
    x_test = np.array(x_test, dtype=np.float32) / 255
    y_test = np.array(y_test, dtype=np.float32) / 255

    print("Obtaining the train generator...")
    # Once we have obtained the dataset division, we create the generators
    train_generator = get_generator(x=x_train,
                                    y=y_train,
                                    batch_size=configuration['batch_size'],
                                    seed=configuration['seed'])
    print("Obtaining the validation generator...")
    validation_generator = get_generator(x=x_val,
                                         y=y_val,
                                         batch_size=configuration['batch_size'],
                                         seed=configuration['seed'],
                                         data_generation_arguments={})
    # print("Obtaining the test generator...")
    # test_generator = get_generator(x=x_test,
    #                                y=y_test,
    #                                batch_size=configuration['batch_size'],
    #                                seed=configuration['seed'],
    #                                data_generation_arguments={})

    return x_train, y_train, x_val, y_val, x_test, y_test, train_generator, validation_generator  # , test_generator


def get_generator(batch_size,
                  x,
                  y,
                  seed=1,
                  data_generation_arguments=None):
    if data_generation_arguments is None:
        data_generation_arguments = dict(width_shift_range=0.5,
                                         zoom_range=[0.5, 1],
                                         horizontal_flip=True,
                                         fill_mode='reflect')
    x_data_generator = ImageDataGenerator(**data_generation_arguments)
    y_data_generator = ImageDataGenerator(**data_generation_arguments, preprocessing_function=last_preprocessing)

    x_augmented = x_data_generator.flow(x, batch_size=batch_size, seed=seed)
    y_augmented = y_data_generator.flow(y, batch_size=batch_size, seed=seed)

    return zip(x_augmented, y_augmented)


def last_preprocessing(image):
    image[image > 0.2] = 1
    image[image <= 0.2] = 0
    return image


class ImageMaskGenerator(Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
