from tensorflow.keras import backend as k
import numpy as np
from metrics import dc
import tensorflow.keras as tf

import dill



def _dice(y_true, y_pred):
    """ Binary dice indice adapted to keras tensors """

    flat_y_true = k.flatten(y_true)
    flat_y_pred = k.flatten(y_pred)

    intersect = k.sum(flat_y_true * flat_y_pred, axis=-1)

    s_true = k.sum(flat_y_true, axis = -1)
    s_pred = k.sum(flat_y_pred, axis = -1)

    return (2. * intersect + 1.) / (s_true + s_pred + 1.)


def multiclass_dice(y_true, y_pred):
    """Extension of the dice to a 3 class problem"""
    """Do not consider class background (0) """
    """ Only DSC healthy and DSC lesion """

    res = k.variable(0., name='dice_classes')

    res_back = _dice(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    res_lesion = _dice(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    # res_healthy = _dice(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    # res_lesion = _dice(y_true[:, :, :, 2], y_pred[:, :, :, 2])

    res = res + 90*res_lesion + 10*res_back
    #res = res + 80*res_lesion + 10*res_healthy + 10*res_back
    #res = res + 33*res_lesion + 33*res_healthy + 33*res_back

    return res / 99


def classes_dice_loss(y_true, y_pred):
    """Loss based on the dice : scales between [0, 1], optimized when minimized"""
    return 1 - multiclass_dice(y_true, y_pred)


