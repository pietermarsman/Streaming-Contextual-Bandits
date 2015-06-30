import random
from scipy.special._ufuncs import expit
import numpy as np

from misc import add_dict

__author__ = 'pieter'

random_coef = lambda: (random.random() - .5) * 0.1

def streaming_lr(t, x_t, y_t, betas, alpha, mu, invscaling=False):
    if betas is None:
        if len(x_t.shape) > 0:
            betas = np.zeros(x_t.shape[0])
        else:
            betas = np.array(0.)
    p = expit(betas.dot(x_t))
    alpha = alpha / pow(t, invscaling) if invscaling else alpha
    betas *= (1 - 2. * alpha * mu)
    betas += alpha * (y_t - p) * x_t
    return betas, alpha


def lr_predict(x, weights, i, learnrate, regulizer, default_value=None):
    """
    Predict value using logistic regression
    :param t:
    :param x:
    :param param:
    :return:
    """
    if default_value is None:
        default_value = lambda: None
    value = weights.get("intercept", default_value())
    for key in x:
        value += weights.get(key, default_value()) * x[key]
    value *= (1. - 2. * learnrate * regulizer) ** i
    return expit(value)


def lsr_update(t, y, x, weights, learnrate, default_value=None):
    """
    Update values using logistic streaming regression
    :param t: target
    :param y: output
    :param x: input
    :param weights:
    :return:
    """
    if default_value is None:
        default_value = lambda: None
    error = t - y
    add_dict(weights, "intercept", learnrate * error, default_value())
    for key in x:
        add_dict(weights, key, learnrate * error * x[key], default_value())
    return weights