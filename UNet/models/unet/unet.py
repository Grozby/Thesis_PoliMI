#
# UNet, with batch normalization, dropout.
# Two main models: w/ attention, w/o attention.
#

from tensorflow.keras import Model, layers

from UNet.metrics_and_losses import metrics
from UNet.models.unet.blocks import convolutional2_d_bn, convolution_block, gating_signal, attention_block, \
    decoder_block


def unet(**configuration):
    # Starting input layer
    input_layer = layers.Input((configuration["input_shape_y"],
                                configuration["input_shape_x"],
                                configuration["input_channels"]))
    x = input_layer
    down_convolution_layers = []
    convolutional_filters_number = configuration["filters"]

    # # # # # # # # # # # #
    #  Unet: Encoder
    # # # # # # # # # # # #

    for n in range(configuration["number_of_layers"]):
        # Convolution block
        x = convolution_block(filters=convolutional_filters_number,
                              batch_normalization=configuration["batch_normalization"],
                              dropout=configuration["dropout"])(x)
        down_convolution_layers.append(x)

        # MaxPooling block
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pooling_{0}".format(n + 1))(x)

        # Update the number of the filters
        convolutional_filters_number = convolutional_filters_number * 2

    # # # # # # # # # # # #
    #  Unet: Bottleneck
    # # # # # # # # # # # #

    x = convolution_block(filters=convolutional_filters_number,
                          batch_normalization=configuration["batch_normalization"],
                          dropout=configuration["dropout"])(x)

    # # # # # # # # # # # #
    #  Unet: Decoder
    # # # # # # # # # # # #

    for n in range(configuration["number_of_layers"]):
        # Update the number of the filters
        convolutional_filters_number = convolutional_filters_number // 2
        x = decoder_block(x, down_convolution_layers, n, convolutional_filters_number, **configuration)

    # # # # # # # # # # # #
    # Output block
    # # # # # # # # # # # #

    x = convolutional2_d_bn(filters=1,
                            kernel_size=1,
                            batch_normalization=configuration["batch_normalization"],
                            activation_layer="sigmoid",
                            kernel_initializer='glorot_uniform')(x)

    # # # # # # # #
    #  Unet: model
    # # # # # # # #

    model = Model(inputs=[input_layer], outputs=[x])

    return model


