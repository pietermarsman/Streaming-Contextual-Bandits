import random
import threading
import time
import datetime

import numpy as np

from agents import ThompsonLogisticAgent
from agents import GreedyAgent
# from agents import RegRegressionAgent
from agents import MultiBetaAgent
from agents import RandomAgent
from communication import get_context, propose_page
from misc import create_directory, add_dict
import matplotlib.pyplot as plt

__author__ = 'pieter'


class Experiment(threading.Thread):
    MAX_I = 10000

    def __init__(self, agent, name=None, run_idx=[0]):
        super().__init__()
        self.agent = agent
        self.run_idx = run_idx
        self.data = dict()
        self.name = str(time.time()).replace(".", "") if name is None else name
        self.time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    def run(self):
        for run_id in self.run_idx:
            for j in range(self.MAX_I + 1):
                context = get_context(run_id, j)
                action = self.agent.decide(context)
                result = propose_page(run_id, j, **action)
                self.agent.feedback(result)
                add_dict(self.data, run_id, [{'context': context, 'action': action, 'result': result}], [])
                success = "Success!" if result["effect"]["Success"] else ""
                print(self.to_string(action, run_id, j, success))
            add_dict(self.data, "reward", self.agent.cum_reward / self.agent.i)
            add_dict(self.data, "cum_reward", self.agent.cum_reward)
            self.save()

    def save(self):
        create_directory("log")
        create_directory("agents")
        time_str = self.time_str + "_"
        np.save('log/' + time_str + self.name, self.data)
        np.save('agents/' + time_str + self.agent.name, self.agent.to_saveable())

    def to_string(self, action, run_id, i, success):
        return "runid={}, i={}, agent={}, reward={:.4f}, action={} {}".format(run_id, i, self.agent.name,
                                                                              self.agent.cum_reward / (i + 1), action,
                                                                              success)


if __name__ == "__main__":
    experiments = []
    Experiment.MAX_I = 10
    for i in range(1):
        runid = random.choice(range(9900))
        str_runid = str(runid).zfill(4)
        # Greedy agent
        greedy_name = "greedy_runid_" + str(runid).zfill(4)
        greedy_agent = GreedyAgent(greedy_name)
        exp_greedy = Experiment(greedy_agent, greedy_name, run_idx=[runid])
        experiments.append(exp_greedy)
        exp_greedy.start()
        # Random agent
        # random_name = "random_runid_" + str(runid).zfill(4)
        # random_agent = RandomAgent(random_name)
        # exp_random = Experiment(random_agent, random_name, run_idx=[runid])
        # exp_random.start()
        # Multi beta agent
        multib_name = "multibeta_runid_" + str(runid).zfill(4)
        multib_agent = MultiBetaAgent(multib_name)
        exp_multib = Experiment(multib_agent, multib_name, run_idx=[runid])
        experiments.append(exp_multib)
        exp_multib.start()
        # Regularized regression agent
        regreg_name = "regreg_runid_" + str(runid).zfill(4)
        regreg_agent = RegRegressionAgent(regreg_name)
        exp_regreg = Experiment(regreg_agent, regreg_name, run_idx=[runid])
        experiments.append(exp_regreg)
        exp_regreg.start()
        # Naive Bayesian agent
        # nb_name = "nb_runid_" + str(runid).zfill(4)
        # nb_agent = NaiveBayesAgent(nb_name)
        # exp_nb = Experiment(nb_agent, nb_name, run_idx=[runid])
        # experiments.append(exp_nb)
        # exp_nb.start()
        # Bootstrap Thompson sampling poor man's Bayes streaming logistic regression
        thomp_name = "thomp_runid_%s" % (str_runid)
        thomp_agent = ThompsonLogisticAgent(thomp_name, 0.01, 1e-3, 200, 100)
        exp_thomp = Experiment(thomp_agent, thomp_name, run_idx=[runid])
        experiments.append(exp_thomp)
        exp_thomp.start()
    while any(map(lambda x: x.is_alive(), experiments)):
        time.sleep(10)
    for experiment in experiments:
        print(experiment.data["reward"], experiment.name)


