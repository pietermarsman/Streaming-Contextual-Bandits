from abc import ABCMeta, abstractmethod
import random

import numpy as np
from sklearn.linear_model import SGDRegressor

from misc import logit
import matplotlib.pyplot as plt

__author__ = 'pieter'

# idea Thomson sampling for simple contextual bandit problems (Frank)
# idea Gibs sampling for probit models (Frank)


class Agent(metaclass=ABCMeta):
    HEADERS = [5, 15, 35]
    ADTYPES = ["skyscraper", "square", "banner"]
    COLORS = ["green", "blue", "red", "black", "white"]
    PRODUCTIDS = range(10, 25)
    PRICES = np.arange(1.0, 50.0, 1.).tolist()

    LANGUAGES = ["EN", "GE", "NL"]
    REFERES = ["Google", "Bing", "NA"]
    AGENTS = ["Windows", "Linux", "OSX", "mobile"]

    def __init__(self, name, saveable=None):
        self.cum_reward = 0
        self.last_action = None
        self.last_success = None
        self.last_reward = None
        self.name = name

    @abstractmethod
    def from_saveable(self, saveable):
        pass

    @abstractmethod
    def decide(self, context):
        pass

    def feedback(self, result):
        self.last_success = result["effect"]["Success"]
        self.last_reward = self.last_success * (self.last_action["price"] if self.last_action is not None else 1)
        self.cum_reward += self.last_reward

    @abstractmethod
    def to_saveable(self):
        pass

    def action_sizes(self):
        return len(Agent.HEADERS), len(Agent.ADTYPES), len(Agent.COLORS), len(Agent.PRODUCTIDS), len(Agent.PRICES)

    @staticmethod
    def generate_action_matrix():
        action_mat = []
        prices = []
        action_values = []
        for header in Agent.HEADERS:
            for adtype in Agent.ADTYPES:
                for color in Agent.COLORS:
                    for product_id in Agent.PRODUCTIDS:
                        for price in Agent.PRICES:
                            action = {"header": header, "adtype": adtype, "color": color, "productid": product_id,
                                      "price": price}
                            action_mat.append(Agent.action_to_vector(action))
                            action_values.append(action)
                            prices.append(price)
        action_mat = np.array(action_mat)
        prices = np.array(prices)
        return action_mat, action_values, prices

    @staticmethod
    def action_to_vector(action):
        header_vector = Agent.one_one_array(len(Agent.HEADERS), Agent.HEADERS.index(action["header"]))
        adtype_vector = Agent.one_one_array(len(Agent.ADTYPES), Agent.ADTYPES.index(action["adtype"]))
        color_vector = Agent.one_one_array(len(Agent.COLORS), Agent.COLORS.index(action["color"]))
        product_vector = Agent.one_one_array(len(Agent.PRODUCTIDS), Agent.PRODUCTIDS.index(action["productid"]))
        price_vector = action["price"] / 50
        return np.hstack((header_vector, adtype_vector, color_vector, product_vector, price_vector))

    @staticmethod
    def one_one_array(size, one_index):
        arr = np.zeros(size)
        arr[one_index] = 1.
        return arr

    @staticmethod
    def context_to_vector(context):
        context = context["context"]
        intercept_vector = np.ones(1)
        age_vector = (context["Age"] - 32) / 10
        language_vector = Agent.one_one_array(len(Agent.LANGUAGES), Agent.LANGUAGES.index(context["Language"]))
        referer_vector = Agent.one_one_array(len(Agent.REFERES), Agent.REFERES.index(context["Referer"]))
        agent_vector = Agent.one_one_array(len(Agent.AGENTS), Agent.AGENTS.index(context["Agent"]))
        return np.hstack((intercept_vector, age_vector, language_vector, referer_vector, agent_vector))

    @staticmethod
    def vector_str():
        return [x for l in [["Intercept", "Age"], Agent.LANGUAGES, Agent.REFERES, Agent.AGENTS, Agent.HEADERS, Agent.ADTYPES,
                  Agent.COLORS, Agent.PRODUCTIDS, ["Price"]] for x in l]


class RandomAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def from_saveable(self, saveable):
        pass

    def decide(self, context):
        header = random.choice(Agent.HEADERS)
        adtype = random.choice(Agent.ADTYPES)
        color = random.choice(Agent.COLORS)
        product_id = random.choice(Agent.PRODUCTIDS)
        price = random.choice(Agent.PRICES)
        self.last_action = {"header": header, "adtype": adtype, "color": color, "productid": product_id, "price": price}
        return self.last_action

    def to_saveable(self):
        pass


class GreedyAgent(Agent):
    def __init__(self, name, saveable=None):
        super().__init__(name, saveable)
        num_headers, num_adtypes, num_colors, num_products, num_prices = self.action_sizes()
        self.counts = dict(header=np.ones(num_headers), adtype=np.ones(num_adtypes), color=np.ones(num_colors),
                           product=np.ones(num_products), price=np.ones(num_prices))
        start = 50
        self.revenue = dict(header=np.ones(num_headers) * start, adtype=np.ones(num_adtypes) * start,
                            color=np.ones(num_colors) * start,
                            product=np.ones(num_products) * start, price=np.ones(num_prices) * start)
        self.last_choice = None
        self.from_saveable(saveable)

    def from_saveable(self, saveable):
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
        self.last_choice = dict(header=header_id, adtype=adtype_id, color=color_id, product=product_id, price=price_id)
        self.last_action = {"header": Agent.HEADERS[header_id], "adtype": Agent.ADTYPES[adtype_id],
                            "color": Agent.COLORS[color_id],
                            "productid": Agent.PRODUCTIDS[product_id], "price": Agent.PRICES[price_id]}
        return self.last_action

    def feedback(self, result):
        super().feedback(result)
        for key in self.counts:
            self.counts[key][self.last_choice[key]] += 1
            self.revenue[key][self.last_choice[key]] += self.last_reward

    def to_saveable(self):
        saveable = dict()
        saveable["counts"] = self.counts
        saveable["revenue"] = self.revenue
        return saveable


class MultiBetaAgent(Agent):
    def __init__(self, name, saveable=None):
        super().__init__(name, saveable)
        num_headers, num_adtypes, num_colors, num_products, num_prices = self.action_sizes()
        self.successes = dict(header=np.ones(num_headers), adtype=np.ones(num_adtypes),
                           color=np.ones(num_colors),
                           product=np.ones(num_products), price=np.ones(num_prices))
        self.counts = dict(header=np.ones(num_headers), adtype=np.ones(num_adtypes),
                           color=np.ones(num_colors),
                           product=np.ones(num_products), price=np.ones(num_prices))
        self.last_choice = None
        self.from_saveable(saveable)

    def from_saveable(self, saveable):
        if saveable is not None:
            self.counts = saveable["counts"]

    def to_saveable(self):
        return {"counts": self.counts}

    def decide(self, context):
        def sample_best(key):
            success_sample = []
            for action in range(self.successes[key].shape[0]):
                success_sample.append(np.random.beta(self.successes[key][action], self.counts[key][action]))
            if key is 'price':
                success_sample = np.array(success_sample) * np.array(Agent.PRICES)
            return np.argmax(success_sample)

        header_id = sample_best("header")
        adtype_id = sample_best("adtype")
        color_id = sample_best("color")
        product_id = sample_best("product")
        price_id = sample_best("price")

        self.last_choice = dict(header=header_id, adtype=adtype_id, color=color_id, product=product_id, price=price_id)
        self.last_action = {"header": Agent.HEADERS[header_id], "adtype": Agent.ADTYPES[adtype_id],
                            "color": Agent.COLORS[color_id],
                            "productid": Agent.PRODUCTIDS[product_id], "price": Agent.PRICES[price_id]}
        return self.last_action

    def feedback(self, result):
        super().feedback(result)
        for key in self.counts:
            self.counts[key][self.last_choice[key]] += 1
            self.successes[key][self.last_choice[key]] += self.last_success


class NaiveLogisticAgent(Agent):
    def __init__(self, name, saveable=None, lambda_=0.05, mu=0.0):
        """
        Logistic agent fitting the function y = logit(Betas * x). y is wheter or not the experiment is a succes. Betas
        are weights for the input variables. x are the input variables; both context and action values. The best action
        is chosen respectively to the expected reward y * price.

        :param name: of the agent
        :param saveable: possible saved instance of the agent
        :param lambda_: step size
        :param mu: regularization coefficient
        :return:
        """
        super().__init__(name, saveable)
        # Saving data for plotting
        self.success_pred = []
        self.failure_pred = []
        # Used for logistic regression
        self.last_context_vec = None
        self.last_action_vec = None
        self.betas = None
        self.lam = lambda_
        self.mu = mu
        # Generating all possible actions in advance
        self.action_mat, self.action_values, self.prices = Agent.generate_action_matrix()

    def from_saveable(self, saveable):
        self.betas = saveable["betas"]

    def to_saveable(self):
        return {"betas": self.betas}

    def decide(self, context):
        self.last_context_vec = self.context_to_vector(context)

        if self.betas is not None:
            context_predictors = np.repeat(self.last_context_vec.reshape((1, -1)), self.action_mat.shape[0], 0)
            predictors = np.hstack((context_predictors, self.action_mat))
            p = logit(predictors.dot(self.betas.reshape((-1, 1))))
            rewards = p * self.prices.reshape((-1, 1))
            norm_rewards = rewards / np.sum(rewards)
            actiond_id = np.where(np.cumsum(norm_rewards) > np.random.random())[0][0]
            self.last_action = self.action_values[actiond_id]
        else:
            self.last_action = random.choice(self.action_values)
        self.last_action_vec = self.action_to_vector(self.last_action)
        return self.last_action

    def feedback(self, result):
        super().feedback(result)
        vector = np.hstack((self.last_context_vec, self.last_action_vec))
        self.streaming_lr(vector, np.array([self.last_success]))

    def streaming_lr(self, x_t, y_t):
        if self.betas is None:
            self.betas = np.zeros(x_t.shape[0])
        p = logit(self.betas.dot(x_t))
        self.betas *= (1 - 2. * self.lam * self.mu)
        self.betas += self.lam * (y_t - p) * x_t

    def __str__(self):
        names = [x for l in
                 [["Intercept", "Age"], Agent.LANGUAGES, Agent.REFERES, Agent.AGENTS, Agent.HEADERS, Agent.ADTYPES,
                  Agent.COLORS, Agent.PRODUCTIDS, ["Price"]] for x in l]
        tuple_list = list(zip(self.betas.tolist(), names))
        formated_list = ", ".join(list(map(lambda x: str(x[1]) + "={:.2f}".format(x[0]), tuple_list)))
        return "Online Logistic Regression with betas: " + str(formated_list)


class RegRegressionAgent(Agent):

    def __init__(self, name, saveable=None):
        super().__init__(name, saveable)
        self.model = SGDRegressor(warm_start=True, penalty='l1', alpha=1e-4, fit_intercept=False, eta0=0.01, power_t=1/3)

        self.last_vec_context = None
        self.last_vec_action = None

        self.count = 0
        self.coefs = []

        self.action_mat, self.action_values, self.prices = Agent.generate_action_matrix()

    def to_saveable(self):
        pass

    def from_saveable(self, saveable):
        pass

    def input_model(self, predictors):
        if len(predictors.shape) == 1:
            predictors = predictors.reshape((1, -1))
        ret = np.vstack((predictors.T, predictors[:, 1] ** 2, predictors[:, -1] ** 2)).T
        return ret

    def decide(self, context):
        self.last_vec_context = Agent.context_to_vector(context)

        context_predictors = np.repeat(self.last_vec_context.reshape((1, -1)), self.action_mat.shape[0], 0)
        predictors = np.hstack((context_predictors, self.action_mat))
        X = self.input_model(predictors)

        if hasattr(self.model, 'intercept_'):
            y = self.model.predict(X)
            y = (y + np.min(y)) ** 2
            plt.figure(2)
            plt.clf()
            plt.ion()
            plt.plot(y,'*')
            if np.sum(y) > 0:
                y /= np.sum(y)
                action_id = np.where(np.cumsum(y) > random.random())[0][0]
                self.last_action = self.action_values[action_id]
                self.last_vec_action = Agent.action_to_vector(self.last_action)
                return self.last_action

        self.last_action = random.choice(self.action_values)
        self.last_vec_action = Agent.action_to_vector(self.last_action)

        return self.last_action

    def feedback(self, result):
        super().feedback(result)

        predictor = np.hstack((self.last_vec_context, self.last_vec_action))
        X = self.input_model(predictor)

        self.model = self.model.partial_fit(X, np.array([self.last_reward]))

        self.plot()

    def plot(self):
        self.coefs.append(list(self.model.coef_))
        self.count += 1
        if self.count % 50 is 0:
            plt.figure(1)
            plt.cla()
            plt.ion()
            plt.plot(np.array(self.coefs))
            vec_str = Agent.vector_str()
            plt.legend(vec_str + ["Age**2", "Price**2"], 'southeast')
            plt.ylim([-3, 3])
            plt.draw()
            plt.pause(0.0001)