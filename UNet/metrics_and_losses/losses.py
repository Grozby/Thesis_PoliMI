from tensorflow.keras import backend as K

from UNet.metrics_and_losses import metrics


def iou_loss(y_true, y_pred):
    return 1 - metrics.iou(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1 - metrics.dice(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    loss_1 = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_0 = - (1 - y_true) * (alpha * K.pow(y_pred, gamma) * K.log(1 - y_pred))
    return K.mean(loss_0 + loss_1)


def focal_loss_dice(y_true, y_pred):
    return dice_loss(y_true, y_pred) + 0.8 * focal_loss(y_true, y_pred)


def focal_jaccard_loss(y_true, y_pred):
    return iou_loss(y_true, y_pred) + 2.0 * focal_loss(y_true, y_pred)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + K.epsilon()) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + K.epsilon())


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def focal_tversky_dice_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 1.0 * focal_tversky(y_true, y_pred)
