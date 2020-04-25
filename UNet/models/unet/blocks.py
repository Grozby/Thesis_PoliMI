from tensorflow.keras import layers, backend as K


def convolutional2_d_bn(filters,
                        kernel_size,
                        batch_normalization=False,
                        activation_layer="relu",
                        kernel_initializer="he_normal",
                        use_bias=False):
    def layer(x):
        x = layers.Conv2D(filters,
                          kernel_size,
                          padding="same",
                          kernel_initializer=kernel_initializer,
                          use_bias=use_bias)(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=3 if K.image_data_format() == 'channels_last' else 1)(x)

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
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

        x = convolutional2_d_bn(filters=filters,
                                kernel_size=3,
                                batch_normalization=batch_normalization)(x)
        return x

    return layer


def expand_as(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)


def gating_signal(input, batch_norm=False):
    shape_input = K.int_shape(input)
    x = layers.Conv2D(filters=shape_input[3] * 1, kernel_size=(1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization(axis=3 if K.image_data_format() == 'channels_last' else 1)(x)
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

    upsample_psi = expand_as(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(filters=shape_x[3], kernel_size=(1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization(axis=3 if K.image_data_format() == 'channels_last' else 1)(result)
    return result_bn


def decoder_block(x, decoder_layer, n, convolutional_filters_number, **configuration):
    if configuration["attention"] and decoder_layer is not None:
        # Gating signal + attention block
        gating = gating_signal(x, batch_norm=configuration["batch_normalization"])
        decoder_layer = attention_block(decoder_layer,
                                        gating,
                                        convolutional_filters_number)

    if configuration["learnable_upsample"]:
        x = layers.Conv2DTranspose(filters=convolutional_filters_number,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding="same",
                                   use_bias=not configuration["batch_normalization"],
                                   name="upsample_{}".format(n + 1))(x)

        if configuration["batch_normalization"]:
            x = layers.BatchNormalization(axis=3 if K.image_data_format() == 'channels_last' else 1)(x)

        x = layers.Activation('relu')(x)
    else:
        x = layers.UpSampling2D(size=(2, 2))(x)

    if decoder_layer is not None:
        x = layers.concatenate([x, decoder_layer], axis=3)

    # Convolution block
    x = convolution_block(filters=convolutional_filters_number,
                          batch_normalization=configuration["batch_normalization"],
                          dropout=configuration["dropout"])(x)

    return x
