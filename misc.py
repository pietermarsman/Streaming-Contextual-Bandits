import os

__author__ = 'pieter'


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def add_dict(dict, key, value):
    if key in dict:
        dict[key] += value
    else:
        dict[key] = value


def create_key(dict, key, default_value=None):
    if key not in dict:
        dict[key] = default_value

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
