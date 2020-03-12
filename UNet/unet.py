import os
from math import ceil

import numpy
import tensorflow as tf

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils

from UNet import exceptions, losses
from UNet import metrics
from UNet import utils
from UNet.dataset_preparation import prepare_dataset
from UNet.model import get_compiled_model

dirname = os.path.dirname(__file__)


class Unet:
    def __init__(self):
        self.configuration = {}
        self.get_configuration()
        print(self.configuration)
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.train_generator = None
        self.validation_generator = None
        loss = losses.focal_tversky_dice_loss
        optimizer = SGD(lr=self.configuration["learning_rate"],
                        momentum=self.configuration["momentum"],
                        decay=self.configuration["decay"],
                        nesterov=self.configuration["nesterov"])

        self.model = get_compiled_model(loss=loss, optimizer=optimizer, **self.configuration)

    def get_configuration(self):
        with open(os.path.join(dirname, "configuration"), 'r') as config_file:
            parser = None
            try:
                line = config_file.readline()
                line = line.rstrip()
                while line:
                    if line in utils.Parser.type_of_parsers:
                        parser = utils.Parser(line)
                    else:
                        config_option, config_value = line.split('=', 1)
                        self.configuration[config_option] = parser.parse_string(config_value)

                    line = config_file.readline()
                    line = line.rstrip()
            except Exception:  # whatever reader errors you care about
                raise exceptions.BadConfigurationFile()

    def load_data(self):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, \
            self.train_generator, self.validation_generator = prepare_dataset(**self.configuration)

    def train(self):
        model_checkpoint = callbacks.ModelCheckpoint('unet_retina.hdf5',
                                                     monitor=self.configuration["monitor_accuracy"],
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max')
        # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorBoardCallBack = callbacks.TensorBoard(log_dir=logdir,
        #                                             update_freq="epoch",
        #                                             write_graph=False,
        #                                             histogram_freq=1,
        #                                             batch_size=ceil(
        #                                                 self.x_train.shape[0] / self.configuration["batch_size"]))

        early_stopping = callbacks.EarlyStopping(monitor=self.configuration["monitor_accuracy"],
                                                 patience=30,
                                                 verbose=1,
                                                 mode='max')

        self.model.fit(x=(pair for pair in self.train_generator),
                       # steps_per_epoch=self.x_train.shape[0],
                       steps_per_epoch=ceil(self.x_train.shape[0] / self.configuration["batch_size"]),
                       epochs=self.configuration["epochs"],
                       validation_data=(pair for pair in self.validation_generator),
                       validation_steps=ceil(self.x_val.shape[0] / self.configuration["batch_size"]),
                       callbacks=[early_stopping,
                                  model_checkpoint,
                                  ],
                       verbose=0)

    def predict_images(self, images):
        return self.model.predict(images, verbose=1)

    def predict_test_images(self):
        return self.predict_images_from_set(self.x_test, self.y_test)

    def predict_validation_images(self):
        return self.predict_images_from_set(self.x_val, self.y_val)

    def predict_images_from_set(self, x, y):
        x_splitted = numpy.array_split(x, 4)
        predictions = []
        for x_s in x_splitted:
            predictions.extend(self.model.predict(x_s, verbose=1))

        ious = metrics.iou_all_value(y, predictions)

        print("Average IoU: {0}".format(numpy.average(ious)))
        print("Standard Deviation IoU: {0}".format(numpy.std(ious)))

        numpy.savetxt("ious.csv", numpy.array(ious), delimiter=",")

        utils.save_result(os.path.join(dirname, "results/predictions"), predictions)
        utils.save_result(os.path.join(dirname, "results/images"), x)
        utils.save_result(os.path.join(dirname, "results/labels"), y)
        utils.plot_images(org_imgs=x,
                          mask_imgs=y,
                          pred_imgs=numpy.array(predictions),
                          nm_img_to_plot=x.shape[0],
                          figsize=6)

    def test_training_images(self):
        for pair in self.train_generator:
            utils.plot_images(pair[0], pair[1], pair[1], show=True, save=False)

    def load_weights(self, file="unet_retina.hdf5"):
        self.model.load_weights(os.path.join(dirname, file))


def main():
    unet = Unet()
    unet.load_data()
    unet.load_weights(file="4layers.hdf5")
    # unet.test_training_images()
    # unet.train()
    unet.predict_test_images()
    # unet.predict_validation_images()


if __name__ == "__main__":
    main()
