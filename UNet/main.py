import datetime
import os
from math import ceil

import numpy
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from UNet import exceptions
from UNet import utils as utilities
from UNet.dataset_preparation import prepare_dataset
from UNet.metrics_and_losses import losses, metrics
from UNet.models.unet.unet_backbone import Unet_EfficientNet

dirname = os.path.dirname(__file__)


class Unet:
    def __init__(self, name, model_factory):
        self.name = name
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
        self.test_generator = None
        loss = losses.dice_loss
        optimizer = SGD(lr=self.configuration["learning_rate"],
                        momentum=self.configuration["momentum"],
                        decay=self.configuration["decay"],
                        nesterov=self.configuration["nesterov"])

        self.model = self.get_compiled_model(model_factory=model_factory,
                                             loss=loss,
                                             optimizer=optimizer,
                                             **self.configuration)

    @staticmethod
    def get_compiled_model(model_factory, loss, optimizer, **configuration):
        model = model_factory(**configuration)

        # Model compilation with apposite optimizer, losses and metrics_and_losses
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metrics.iou],
            experimental_run_tf_function=False)

        return model

    def get_configuration(self):
        with open(os.path.join(dirname, "configuration"), 'r') as config_file:
            parser = None
            try:
                line = config_file.readline()
                line = line.rstrip()
                while line:
                    if line in utilities.Parser.type_of_parsers:
                        parser = utilities.Parser(line)
                    else:
                        config_option, config_value = line.split('=', 1)
                        self.configuration[config_option] = parser.parse_string(config_value)

                    line = config_file.readline()
                    line = line.rstrip()
            except Exception:  # whatever reader errors you care about
                raise exceptions.BadConfigurationFile()

    def load_data(self):
        self.x_train, self.y_train, \
        self.x_val, self.y_val, \
        self.x_test, self.y_test, \
        self.train_generator, \
        self.validation_generator, \
        self.test_generator = prepare_dataset(**self.configuration)

    def train(self):
        if "path_to_drive" not in self.configuration.keys():
            self.configuration["path_to_drive"] = ""

        model_checkpoint = callbacks.ModelCheckpoint('{}{}.hdf5'.format(self.configuration["path_to_drive"],
                                                                        self.name),
                                                     monitor=self.configuration["monitor_accuracy"],
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max')
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        early_stopping = callbacks.EarlyStopping(monitor=self.configuration["monitor_accuracy"],
                                                 patience=20,
                                                 verbose=1,
                                                 mode='max')

        print("Start fit...")
        self.model.fit(x=self.train_generator,
                       # steps_per_epoch=self.x_train.shape[0],
                       # steps_per_epoch=self.x_train.shape[0],
                       epochs=self.configuration["epochs"],
                       validation_data=self.validation_generator,
                       # validation_steps=self.x_val.shape[0],
                       callbacks=[early_stopping,
                                  model_checkpoint,
                                  tensorboard_callback
                                  ],
                       verbose=1)

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

        utilities.save_result(os.path.join(dirname, "results/predictions"), predictions)
        utilities.save_result(os.path.join(dirname, "results/images"), x)
        utilities.save_result(os.path.join(dirname, "results/labels"), y)
        utilities.plot_images(org_imgs=x,
                              mask_imgs=y,
                              pred_imgs=numpy.array(predictions),
                              nm_img_to_plot=x.shape[0],
                              figsize=6)

    def test_training_images(self):
        for pair in self.validation_generator:
            print("boss")
            # utilities.plot_images(pair[0], numpy.average(pair[1], axis=3), numpy.average(pair[1], axis=3), show=True,
            #                       save=False)

    def load_weights(self, file="unet_retina.hdf5"):
        self.model.load_weights(os.path.join(dirname, file))


def main():
    unet = Unet(name="Unet_EfficientNetB4", model_factory=Unet_EfficientNet)
    unet.load_data()
    # tf.keras.utils.plot_model(unet.model, show_shapes=True)
    # unet.load_weights()
    # unet.load_weights(file="4layers.hdf5")
    unet.test_training_images()
    # unet.train()
    # unet.predict_test_images()
    # unet.predict_validation_images()


if __name__ == "__main__":
    main()
