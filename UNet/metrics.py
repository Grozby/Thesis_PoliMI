from tensorflow.keras import backend as K


def iou_not_correct(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (intersection + K.epsilon()) / (union - intersection + K.epsilon())


def dice_not_correct(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2. * intersection + K.epsilon()) / (union + K.epsilon())


def get_union_and_intersection(y_true, y_pred):
    # We flatten the two prediction tensor in a 2D shape,
    # where the first axis is specify the prediction image,
    # the second axis will be the image's flatten pixels.
    y_true_batch_flatten = K.batch_flatten(y_true)
    y_pred_batch_flatten = K.batch_flatten(y_pred)

    # We sum over the second axis! In this way, we obtain the intersection value
    # for the pair ground-truth / prediction. Same for the union.
    intersection = K.sum(y_true_batch_flatten * y_pred_batch_flatten, axis=1, keepdims=True)
    union = K.sum(y_true_batch_flatten, axis=1, keepdims=True) + \
            K.sum(y_pred_batch_flatten, axis=1, keepdims=True)

    # We return 1D vectors, in which the ith position contains the union/intersection value
    # for the image in the ith position in the batch.
    return union, intersection


def iou(y_true, y_pred):
    union, intersection = get_union_and_intersection(y_true, y_pred)
    return K.mean((intersection + K.epsilon()) / (union - intersection + K.epsilon()))


def iou_all_value(y_true, y_pred):
    union, intersection = get_union_and_intersection(y_true, y_pred)
    return (intersection + K.epsilon()) / (union - intersection + K.epsilon())


def dice(y_true, y_pred):
    union, intersection = get_union_and_intersection(y_true, y_pred)
    return K.mean(((2. * intersection) + K.epsilon()) / (union + K.epsilon()))


def jaccard_coefficient(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + K.epsilon()) / (union - intersection + K.epsilon())
    return K.mean(jac)




