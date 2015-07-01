from abc import ABCMeta, abstractmethod
import random
import sys
import threading

import numpy as np
from scipy.special._ufuncs import expit
# from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

from misc import plot_continuously, create_key, add_dict




# Super action: {'adtype': 'skyscraper', 'productid': 10, 'header': 5, 'color': 'green', 'price': 49.0}
from regression import lr_predict, lsr_update
import regression

__author__ = 'pieter'

# idea Thomson sampling for simple contextual bandit problems (Frank)
# idea Gibs sampling for probit models (Frank)

EPS = sys.float_info.epsilon


class Agent(metaclass=ABCMeta):
    HEADERS = [5, 15, 35]
    ADTYPES = ["skyscraper", "square", "banner"]
    COLORS = ["green", "blue", "red", "black", "white"]
    PRODUCTIDS = range(10, 25)
    PRICES = np.arange(1.0, 50.0, 5.).tolist()

    LANGUAGES = ["EN", "GE", "NL"]
    REFERES = ["Google", "Bing", "NA"]
    AGENTS = ["Windows", "Linux", "OSX", "mobile"]

    def __init__(self, name, saveable=None):
        self.cum_reward = 0
        self.last_action = None
        self.last_success = None
        self.last_reward = None
        self.name = name
        self.i = 0

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
        self.i += 1

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
        return [x for l in
                [["Intercept", "Age"], Agent.LANGUAGES, Agent.REFERES, Agent.AGENTS, Agent.HEADERS, Agent.ADTYPES,
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


class ThompsonLogisticAgent(Agent):
    def __init__(self, name, learnrate, regulizer, lr_n, action_n, saveable=None, prior=None):
        super().__init__(name, saveable)
        self.learnrate = learnrate
        self.regulizer = regulizer
        self.action_n = action_n
        _, self.actions, _ = Agent.generate_action_matrix()
        self.lrs = [{} for _ in range(lr_n)]
        if prior is not None:
            prior_n = min(lr_n, len(prior))
            self.lrs[:prior_n] = prior[:prior_n]
        self.last_context = None
        self.lrs_lock = threading.Lock()
        self.from_saveable(saveable)

    def decide(self, context):
        self.last_context = context["context"]
        lr = random.sample(self.lrs, 1)[0]
        best_value = -1e100
        self.last_action = None
        actions = random.sample(self.actions, self.action_n)
        xs = map(lambda x: ThompsonLogisticAgent.design(self.last_context, x), actions)
        for x_i, x in enumerate(xs):
            # x = ThompsonLogisticAgent.design(self.last_context, action)
            p = lr_predict(x, lr, self.i, self.learnrate, self.regulizer, regression.random_coef)
            value = p * actions[x_i]['price']
            if value > best_value:
                best_value = value
                self.last_action = actions[x_i]
        return self.last_action

    def feedback(self, result):
        super().feedback(result)
        x = ThompsonLogisticAgent.design(self.last_context, self.last_action)
        for i in range(len(self.lrs)):
            if random.random() > .5:
                p = lr_predict(x, self.lrs[i], self.i, self.learnrate, self.regulizer, regression.random_coef)
                with self.lrs_lock:
                    self.lrs[i] = lsr_update(self.last_success, p, x, self.lrs[i], self.learnrate, regression.random_coef)
                    self.lrs[i] = lsr_update(self.last_success, p, x, self.lrs[i], self.learnrate, regression.random_coef)

    def from_saveable(self, saveable):
        if saveable is not None:
            self.lrs = saveable['lrs']
            self.learnrate = saveable['learnrate']
            self.regulizer = saveable['regulizer']
            self.action_n = saveable['action_n']

    def to_saveable(self):
        return {'lrs': self.lrs, 'learnrate':self.learnrate, 'regulizer':self.regulizer, 'action_n':self.action_n}

    def map_lr(self):
        lr_values = dict()
        with self.lrs_lock:
            for lr in self.lrs:
                for key, value in lr.items():
                    add_dict(lr_values, key, [value * (1. - 2. * self.learnrate * self.regulizer) ** self.i])
        return lr_values

    def plot(self, include=None, exclude=None):
        lr_values = self.map_lr()
        if len(lr_values) > 0:
            keys = sorted(lr_values.keys())
            if include is not None:
                keys = [k for k in keys if any([incl in k for incl in include])]
            if exclude is not None:
                keys = [k for k in keys if all([excl not in k for excl in exclude])]
            plt.ion()
            plt.cla()
            plt.title("Iteration %d" % self.i)
            plt.boxplot([lr_values[k] for k in keys], widths=0.5, showmeans=True)
            plt.xticks(range(1, len(keys)), keys, rotation='vertical')
            plt.subplots_adjust(left=0.05, right=1.0, top=0.95, bottom=0.3)
            plt.draw()
            plt.pause(0.001)

    @staticmethod
    def design(context, action):
        def price(p):
            return (p - 25.) / 25.

        def age(a):
            return (a - 32.) / 32.

        temp = {}
        if context is not None:
            temp.update(context)
        temp.update(action)
        temp['price'] = price(temp['price'])
        if context is not None:
            temp['Age'] = age(temp['Age'])
        x = {}
        done = {}
        for k1, v1 in temp.items():
            cont1 = type(v1) == float
            name1 = str(k1) if cont1 else "%s=%s" % (k1, str(v1))
            value1 = v1 if cont1 else 1
            x[name1] = value1
            for k2, v2 in temp.items():
                if k2 not in done:
                    cont2 = type(v2) == float
                    name2 = str(k2) if cont2 else "%s=%s" % (k2, str(v2))
                    value2 = v2 if cont2 else 1
                    x["%s_%s" % (name1, name2)] = value1 * value2
            done[k1] = True
        return x

    @staticmethod
    def parse_priors(files):
        agents = [np.load(file).item() for file in files]
        return [agent['lrs'][0] for agent in agents if 'lrs' in agent]


# class RegRegressionAgent(Agent):
#     def __init__(self, name, saveable=None):
#         super().__init__(name, saveable)
#         self.model = SGDRegressor(warm_start=True, penalty='l1', alpha=1e-4, fit_intercept=False, eta0=0.01,
#                                   power_t=1 / 3)

#         self.last_vec_context = None
#         self.last_vec_action = None

#         self.count = 0
#         self.coefs = []

#         self.action_mat, self.action_values, self.prices = Agent.generate_action_matrix()

#     def to_saveable(self):
#         pass

#     def from_saveable(self, saveable):
#         pass

#     def input_model(self, predictors):
#         if len(predictors.shape) == 1:
#             predictors = predictors.reshape((1, -1))
#         ret = np.vstack((predictors.T, predictors[:, 1] ** 2, predictors[:, -1] ** 2)).T
#         return ret

#     def decide(self, context):
#         self.last_vec_context = Agent.context_to_vector(context)

#         context_predictors = np.repeat(self.last_vec_context.reshape((1, -1)), self.action_mat.shape[0], 0)
#         predictors = np.hstack((context_predictors, self.action_mat))
#         X = self.input_model(predictors)
#         self.last_action = random.choice(self.action_values)
#         self.last_vec_action = Agent.action_to_vector(self.last_action)

#         return self.last_action

#     def feedback(self, result):
#         super().feedback(result)

#         predictor = np.hstack((self.last_vec_context, self.last_vec_action))
#         X = self.input_model(predictor)

#         self.model = self.model.partial_fit(X, np.array([self.last_reward]))

#     def plot(self):
#         self.coefs.append(list(self.model.coef_))
#         self.count += 1
#         if self.count % 50 is 0:
#             plt.figure(1)
#             plt.cla()
#             plt.ion()
#             plt.plot(np.array(self.coefs))
#             vec_str = Agent.vector_str()
#             plt.legend(vec_str + ["Age**2", "Price**2"], 'southeast')
#             plt.ylim([-3, 3])
#             plt.draw()
#             plt.pause(0.0001)


# class NaiveBayesAgent(Agent):
#     def __init__(self, name, saveable=None, lambda_=0.05, mu=0.0):
#         super().__init__(name, saveable)
#
#         self.lambda_ = lambda_
#         self.mu = mu
#
#         self.mat_action, self.action_values, self.prices = Agent.generate_action_matrix()
#         self.last_vec_context = None
#         self.last_vec_action = None
#
#         self.names = Agent.vector_str()
#         self.successes = np.ones(self.num_features()) * 5
#         self.failures = np.ones(self.num_features())
#         self.beta_calculations = dict()
#         self.weights = np.hstack((np.ones((self.num_features(), 1)) * -.5, np.zeros((self.num_features(), 1))))
#
#         self.binary_idx = range(self.num_features())
#         self.cont_idx = list(map(lambda x: x % self.num_features(), [-1, -2]))
#         self.binary_idx = [i for i in self.binary_idx if i not in self.cont_idx]
#
#         self.count = -1
#
#     def to_saveable(self):
#         pass
#
#     def from_saveable(self, saveable):
#         pass
#
#     def input_model(self, context, actions):
#         if len(context.shape) == 1:
#             context = context.reshape((1, -1))
#         if len(actions.shape) == 1:
#             actions = actions.reshape((1, -1))
#         return np.hstack((actions, actions[:, -1].reshape((-1, 1)) ** 2))
#
#     def num_features(self):
#         return self.mat_action.shape[1] + 1
#
#     def decide(self, context):
#         self.last_vec_context = Agent.context_to_vector(context)
#
#         context_predictors = np.repeat(self.last_vec_context.reshape((1, -1)), self.mat_action.shape[0], 0)
#         X = self.input_model(context_predictors, self.mat_action)
#         # y = expected success probability * price
#         y = np.log(X[:, -2] + EPS)
#
#         for cont_id in self.cont_idx:
#             inp = np.hstack((np.zeros((X.shape[0], 1)), X[:, cont_id].reshape((-1, 1))))
#             # noise = np.random.random((inp.shape[0], 1)) - .5
#             vec = np.vectorize(lambda x: x + 6 * random.random() - 3.)
#             logit_inp = vec(inp.dot(self.weights[cont_id, :]))
#             y += np.log(expit(logit_inp) + EPS)
#
#         for row_i in range(X.shape[0]):
#             for binary_id in self.binary_idx:
#                 if X[row_i, binary_id] == 1:
#                     suc = self.successes[binary_id]
#                     fail = self.failures[binary_id]
#                     create_key(self.beta_calculations, suc, dict())
#                     if fail not in self.beta_calculations[suc]:
#                         self.beta_calculations[suc][fail] = np.log(np.random.beta(suc, fail) + EPS)
#                     y[row_i] += self.beta_calculations[suc][fail]
#
#         best_id = np.argmax(y)
#         self.last_action = self.action_values[best_id]
#         self.last_vec_action = Agent.action_to_vector(self.last_action)
#
#         return self.last_action
#
#     def feedback(self, result):
#         super().feedback(result)
#
#         X = self.input_model(self.last_vec_context, self.last_vec_action)
#
#         for binary_id in self.binary_idx:
#             if X[0, binary_id] == 1:
#                 self.successes[binary_id] += self.last_success
#                 self.failures[binary_id] += not self.last_success
#
#         for cont_id in self.cont_idx:
#             x = np.array([1, X[0, cont_id]])
#             y = vec(inp.dot(self.weights[cont_id, :]))
#             self.weights[cont_id, :] = regression.streaming_lr(self.last_success, x, self.weights[cont_id, :],
#                                                          self.lambda_, self.mu)

        # self.count += 1
        # if self.count % 100 == 0:
        #     plot_continuously(plt.plot, 2, self.successes)
        #     plot_continuously(plt.plot, 2, self.failures, False)
        #     plot_continuously(plt.plot, 2, self.successes / (self.successes + self.failures) * 100, False)
        #     plt.xticks(range(self.num_features()), self.names[12:] + ["Price^2"], rotation="vertical")
        #     for cont_id in self.cont_idx:
        #         plot_continuously(plt.plot, 3, self.weights[cont_id, :], cont_id == self.cont_idx[0])
        #     plt.legend(list(map(str, self.cont_idx)))
        #     plt.draw()
        #     plt.pause(0.0001)
