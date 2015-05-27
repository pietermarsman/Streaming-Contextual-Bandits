from abc import ABCMeta, abstractmethod
import numpy as np
import random

__author__ = 'pieter'


class Agent(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def decide(self, context):
        pass

    @abstractmethod
    def feedback(self, result):
        pass


class RandomAgent(Agent):

    def __init__(self):
        self.headers = [5, 15, 35]
        self.adtype = ["skyscraper", "square", "banner"]
        self.colors = ["green", "blue", "red", "black", "white"]
        self.product_id = range(10, 25)
        self.price = np.arange(0.0, 50.0, 1.)[1:].tolist()

    def decide(self, context):
        header = random.choice(self.headers)
        adtype = random.choice(self.adtype)
        color = random.choice(self.colors)
        product_id = random.choice(self.product_id)
        price = random.choice(self.price)
        return {"header": header, "adtype": adtype, "color":color, "productid": product_id, "price":price}

    def feedback(self, result):
        pass