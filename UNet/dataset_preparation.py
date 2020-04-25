from __future__ import print_function

import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from UNet.data_generators import DataGenerator, DataGeneratorAlbumentations

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
    x, y, x_test, y_test = (np.empty((0, 256, 512, 4), dtype=np.uint8) for _ in range(4))

    for directory in directories:
        images = []
        masks = []
        for (_, _, filenames) in os.walk(path_prediction_images + directory):
            for filename in filenames:
                if ".png" in filename:
                    image = np.array(Image.open(path_training_images + directory + '/' + filename).convert("RGBA"),
                                     dtype=np.uint8)
                    mask = np.array(Image.open(path_prediction_images + directory + '/' + filename).convert("RGBA"),
                                    dtype=np.uint8)
                    # We invert the pixels because in our ground truth images black is where the retina is
                    mask[..., :3] = 255 - mask[..., :3]
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

    print("Obtaining the train generator...")
    train_generator = DataGeneratorAlbumentations(images=x_train,
                                                  labels=y_train,
                                                  batch_size=configuration['batch_size'],
                                                  length=configuration["steps_per_epoch"],
                                                  image_shape=(configuration['input_shape_y'],
                                                               configuration['input_shape_x'],
                                                               configuration['input_channels'])
                                                  )
    print("Obtaining the validation generator...")
    validation_generator = DataGeneratorAlbumentations(images=x_val, labels=y_val,
                                                       batch_size=configuration['batch_size'],
                                                       length=configuration["steps_per_epoch"],
                                                       image_shape=(configuration['input_shape_y'],
                                                                    configuration['input_shape_x'],
                                                                    configuration['input_channels'])
                                                       )
    print("Obtaining the test generator...")
    test_generator = DataGeneratorAlbumentations(images=x_test, labels=y_test,
                                                 batch_size=configuration['batch_size'],
                                                 length=None,
                                                 do_augment=False,
                                                 image_shape=(configuration['input_shape_y'],
                                                              configuration['input_shape_x'],
                                                              configuration['input_channels'])
                                                 )
    print("Obtaining the train dataset...")
    x_train = np.array(x_train, dtype=np.float32) / 255
    y_train = np.array(y_train, dtype=np.float32) / 255
    print("Obtaining the validation dataset...")
    x_val = np.array(x_val, dtype=np.float32) / 255
    y_val = np.array(y_val, dtype=np.float32) / 255
    print("Obtaining the test dataset...")
    x_test = np.array(x_test, dtype=np.float32) / 255
    y_test = np.array(y_test, dtype=np.float32) / 255

    return x_train, y_train, x_val, y_val, x_test, y_test, train_generator, validation_generator, test_generator


def augment_mask(image):
    image[image > 0.2 * 255] = 255
    image[image <= 0.2 * 255] = 0
    return image
