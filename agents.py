from abc import ABCMeta, abstractmethod
import random

import numpy as np


__author__ = 'pieter'


class Agent(metaclass=ABCMeta):

    def __init__(self, name, saveable=None):
        self.headers = [5, 15, 35]
        self.adtypes = ["skyscraper", "square", "banner"]
        self.colors = ["green", "blue", "red", "black", "white"]
        self.product_ids = range(10, 25)
        self.prices = np.arange(0.0, 50.0, 1.)[1:].tolist()
        self.cum_reward = 0
        self.name = name

    def from_saveable(self, saveable):
        if saveable is not None:
            self.name = saveable["name"]

    @abstractmethod
    def decide(self, context):
        pass

    @abstractmethod
    def feedback(self, result):
        pass

    def to_saveable(self):
        return {"name": self.name}


class RandomAgent(Agent):

    def __init__(self, name):
        super().__init__(name)

    def decide(self, context):
        header = random.choice(self.headers)
        adtype = random.choice(self.adtypes)
        color = random.choice(self.colors)
        product_id = random.choice(self.product_ids)
        price = random.choice(self.prices)
        return {"header": header, "adtype": adtype, "color": color, "productid": product_id, "price": price}

    def feedback(self, result):
        pass


class GreedyAgent(Agent):
    def __init__(self, name, saveable=None):
        super().__init__(name, saveable)
        num_headers = len(self.headers)
        num_adtypes = len(self.adtypes)
        num_colors = len(self.colors)
        num_products = len(self.product_ids)
        num_price = len(self.prices)
        self.counts = dict(header=np.ones(num_headers), adtype=np.ones(num_adtypes), color=np.ones(num_colors),
                           product=np.ones(num_products), price=np.ones(num_price))
        start = 500
        self.revenue = dict(header=np.ones(num_headers) * start, adtype=np.ones(num_adtypes) * start, color=np.ones(num_colors) * start,
                              product=np.ones(num_products) * start, price=np.ones(num_price) * start)
        self.last_action = None
        self.last_success = None
        self.last_reward = None
        self.from_saveable(saveable)

    def from_saveable(self, saveable):
        super().from_saveable(saveable)
        if saveable is not None:
            self.counts = saveable["counts"]
            self.revenue = saveable["revenue"]

    def decide(self, context):
        def compute_best(attr):
            return np.argmax(self.revenue[attr] / self.counts[attr])

        header_id = compute_best("header")
        adtype_id = compute_best("adtype")
        color_id = compute_best("color")
        product_id = compute_best("product")
        price_id = compute_best("price")
        self.last_action = dict(header=header_id, adtype=adtype_id, color=color_id, product=product_id, price=price_id)
        return {"header": self.headers[header_id], "adtype": self.adtypes[adtype_id], "color": self.colors[color_id],
                "productid": self.product_ids[product_id], "price": self.prices[price_id]}

    def feedback(self, result):
        self.last_success = result["effect"]["Success"]
        self.last_reward = self.last_success * self.prices[self.last_action["price"]]
        self.cum_reward += self.last_reward
        for key in self.counts:
            self.counts[key][self.last_action[key]] += 1
            self.revenue[key][self.last_action[key]] += self.last_reward

    def to_saveable(self):
        saveable = super().to_saveable()
        saveable["counts"] = self.counts
        saveable["revenue"] = self.revenue
        return saveable