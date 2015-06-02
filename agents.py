from abc import ABCMeta, abstractmethod
import random
import numpy as np
from misc import logit


__author__ = 'pieter'

# idea Dirichlet distribution for better greedy decisions (without context)
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
        num_headers = len(Agent.HEADERS)
        num_adtypes = len(Agent.ADTYPES)
        num_colors = len(Agent.COLORS)
        num_products = len(Agent.PRODUCTIDS)
        num_price = len(Agent.PRICES)
        self.counts = dict(header=np.ones(num_headers), adtype=np.ones(num_adtypes), color=np.ones(num_colors),
                           product=np.ones(num_products), price=np.ones(num_price))
        start = 50
        self.revenue = dict(header=np.ones(num_headers) * start, adtype=np.ones(num_adtypes) * start, color=np.ones(num_colors) * start,
                              product=np.ones(num_products) * start, price=np.ones(num_price) * start)
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
        self.last_action = {"header": Agent.HEADERS[header_id], "adtype": Agent.ADTYPES[adtype_id], "color": Agent.COLORS[color_id],
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


class LogisticAgent(Agent):

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
        self.action_mat = []
        self.action_values = []
        self.prices = []
        for header in Agent.HEADERS:
            for adtype in Agent.ADTYPES:
                for color in Agent.COLORS:
                    for product_id in Agent.PRODUCTIDS:
                        for price in Agent.PRICES:
                            action = {"header": header, "adtype": adtype, "color": color, "productid": product_id, "price": price}
                            self.action_mat.append(self.action_to_vector(action))
                            self.action_values.append(action)
                            self.prices.append(price)
        self.action_mat = np.array(self.action_mat)
        self.prices = np.array(self.prices)

    def from_saveable(self, saveable):
        self.betas = saveable["betas"]

    def to_saveable(self):
        return {"betas": self.betas}

    def decide(self, context):
        self.last_context_vec = self.context_to_vector(context)

        if self.betas is not None:
            predictors = np.hstack((np.repeat(self.last_context_vec.reshape((1, -1)), self.action_mat.shape[0], 0), self.action_mat))
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
        # y_ = logit(vector.dot(self.betas))
        # if self.last_success > 0.:
        #     self.success_pred.append(y_)
        # else:
        #     self.failure_pred.append(y_)
        # if (len(self.success_pred) + len(self.failure_pred)) % 10 == 0:
        #     plt.ion()
        #     plt.cla()
        #     plt.hist([self.success_pred, self.failure_pred], normed=True)
        #     plt.draw()
        #     plt.pause(0.001)

    def streaming_lr(self, x_t, y_t):
        if self.betas is None:
            self.betas = np.zeros(x_t.shape[0])
        p = logit(self.betas.dot(x_t))
        self.betas *= (1 - 2. * self.lam * self.mu)
        self.betas += self.lam * (y_t - p) * x_t

    def __str__(self):
        names = [x for l in [["Intercept", "Age"], Agent.LANGUAGES, Agent.REFERES, Agent.AGENTS, Agent.HEADERS, Agent.ADTYPES, Agent.COLORS, Agent.PRODUCTIDS, ["Price"]] for x in l]
        tuple_list = list(zip(self.betas.tolist(), names))
        formated_list = ", ".join(list(map(lambda x: str(x[1]) + "={:.2f}".format(x[0]), tuple_list)))
        return "Online Logistic Regression with betas: " + str(formated_list)

    @staticmethod
    def one_one_array(size, one_index):
        arr = np.zeros(size)
        arr[one_index] = 1.
        return arr

    @staticmethod
    def action_to_vector(action):
        header_vector = LogisticAgent.one_one_array(len(Agent.HEADERS), Agent.HEADERS.index(action["header"]))
        adtype_vector = LogisticAgent.one_one_array(len(Agent.ADTYPES), Agent.ADTYPES.index(action["adtype"]))
        color_vector = LogisticAgent.one_one_array(len(Agent.COLORS), Agent.COLORS.index(action["color"]))
        product_vector = LogisticAgent.one_one_array(len(Agent.PRODUCTIDS), Agent.PRODUCTIDS.index(action["productid"]))
        price_vector = action["price"] / 50
        return np.hstack((header_vector, adtype_vector, color_vector, product_vector, price_vector))


    @staticmethod
    def context_to_vector(context):
        context = context["context"]
        intercept_vector = np.ones(1)
        age_vector = context["Age"] / 100
        language_vector = LogisticAgent.one_one_array(len(Agent.LANGUAGES), Agent.LANGUAGES.index(context["Language"]))
        referer_vector = LogisticAgent.one_one_array(len(Agent.REFERES), Agent.REFERES.index(context["Referer"]))
        agent_vector = LogisticAgent.one_one_array(len(Agent.AGENTS), Agent.AGENTS.index(context["Agent"]))
        return np.hstack((intercept_vector, age_vector, language_vector, referer_vector, agent_vector))
