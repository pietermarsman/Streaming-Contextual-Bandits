from scipy.special._ufuncs import expit
import numpy as np

__author__ = 'pieter'


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