from tensorflow.keras import layers, Model

from UNet.models.efficientnet.efficientnet import EfficientNetB4
from UNet.models.unet.blocks import convolutional2_d_bn, decoder_block


def freeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def build_unet(
        backbone,
        skip_connection_layers,
        **configuration
):
    # # # # # # # # # # # #
    #  Unet: Encoder
    # # # # # # # # # # # #
    input_ = backbone.input
    x = backbone.output
    convolutional_filters_number = configuration["filters"] * (2 ** configuration["number_of_layers"])

    # extract skip connections
    down_convolution_layers = ([
        backbone.get_layer(name=i).output
        if isinstance(i, str)
        else backbone.get_layer(index=i).output
        for i in skip_connection_layers
    ])

    # # # # # # # # # # # #
    #  Unet: Decoder
    # # # # # # # # # # # #
    for n in range(configuration["number_of_layers"]):
        convolutional_filters_number = convolutional_filters_number // 2
        decoder_layer = down_convolution_layers[n] if n + 1 < len(down_convolution_layers) else None
        x = decoder_block(x, decoder_layer, n, convolutional_filters_number, **configuration)

    # # # # # # # # # # # #
    # Output block
    # # # # # # # # # # # #

    x = convolutional2_d_bn(filters=1,
                            kernel_size=1,
                            activation_layer="sigmoid",
                            kernel_initializer='glorot_uniform',
                            use_bias=True,
                            batch_normalization=False)(x)

    model = Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet_EfficientNet(**configuration):
    backbone = EfficientNetB4(include_top=False,
                              input_shape=(configuration["input_shape_y"],
                                           configuration["input_shape_x"],
                                           configuration["input_channels"]),
                              weights=configuration['efficientnet_weights'])

    encoder_features = ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation')

    model = build_unet(
        backbone=backbone,
        skip_connection_layers=encoder_features,
        **configuration
    )

    # lock encoder weights for fine-tuning
    if configuration["encoder_freeze"]:
        freeze_model(backbone)

    return model
