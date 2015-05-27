import os

__author__ = 'pieter'


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)