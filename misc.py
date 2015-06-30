import os

import matplotlib.pyplot as plt

__author__ = 'pieter'


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def add_dict(dict, key, value, default_value=None):
    if default_value is None:
        dict[key] = dict.get(key, type(value)()) + value
    else:
        dict[key] = dict.get(key, default_value) + value


def create_key(dict, key, default_value=None):
    if key not in dict:
        dict[key] = default_value


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def plot_continuously(func, fig, data, cla=True):
    plt.figure(fig)
    if cla:
        plt.cla()
    plt.ion()
    func(data)
