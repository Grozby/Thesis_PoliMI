from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from UNet import metrics


def convolutional2_d_bn(filters,
                        kernel_size,
                        batch_normalization=False,
                        activation_layer="relu",
                        kernel_initializer="he_normal"):
    def layer(x):
        x = layers.Conv2D(filters,
                          kernel_size,
                          padding="same",
                          kernel_initializer=kernel_initializer,
                          use_bias=not batch_normalization)(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)

        if activation_layer != "lrelu":
            x = layers.Activation(activation_layer)(x)
        else:
            x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    return layer


def convolution_block(filters,
                      batch_normalization=False,
                      dropout=0.0):
    def layer(x):
        x = convolutional2_d_bn(filters=filters,
                                kernel_size=3,
                                batch_normalization=batch_normalization)(x)

        # Dropout if needed
        if dropout <= 0:
            x = layers.Dropout(dropout)(x)

        x = convolutional2_d_bn(filters=filters,
                                kernel_size=3,
                                batch_normalization=batch_normalization)(x)
        return x

    return layer


def expend_as(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)


def gating_signal(input, batch_norm=False):
    shape_input = K.int_shape(input)
    x = layers.Conv2D(filters=shape_input[3] * 1, kernel_size=(1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def attention_block(x, gating, filters):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv2D(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                        padding='same')(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
        sigmoid_xg)

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(filters=shape_x[3], kernel_size=(1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


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

        if configuration["attention"]:
            # Gating signal + attention block

            gating = gating_signal(x, batch_norm=configuration["batch_normalization"])
            attention = attention_block(down_convolution_layers[configuration["number_of_layers"] - n - 1],
                                        gating,
                                        convolutional_filters_number)

            if configuration["learnable_upsample"]:
                x = layers.concatenate([layers.Conv2DTranspose(filters=convolutional_filters_number,
                                                               kernel_size=(3, 3),
                                                               strides=(2, 2),
                                                               padding="same",
                                                               activation="relu",
                                                               name="upsample_{}".format(n + 1))(x),
                                        attention], axis=3)
            else:
                x = layers.concatenate([layers.UpSampling2D(size=(2, 2))(x), attention], axis=3)

            # Convolution block
            x = convolution_block(filters=convolutional_filters_number,
                                  batch_normalization=configuration["batch_normalization"],
                                  dropout=configuration["dropout"])(x)

        else:
            # Convolution block
            x = convolutional2_d_bn(filters=convolutional_filters_number,
                                    kernel_size=2,
                                    batch_normalization=configuration["batch_normalization"])(
                (layers.UpSampling2D(size=(2, 2))(x)))
            x = layers.concatenate([down_convolution_layers[configuration["number_of_layers"] - n - 1], x], axis=3)

            x = convolution_block(filters=convolutional_filters_number,
                                  batch_normalization=configuration["batch_normalization"],
                                  dropout=configuration["dropout"])(x)

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


def get_compiled_model(loss, optimizer, **configuration):
    model = unet(**configuration)

    # Model compilation with apposite optimizer, losses and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metrics.iou],
        experimental_run_tf_function=False)

    return model
