import random
import threading
import time
import datetime

import numpy as np

from communication import get_context, propose_page
from misc import create_directory, add_dict

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
            for i in range(self.MAX_I + 1):
                context = get_context(run_id, i)
                action = self.agent.decide(context)
                result = propose_page(run_id, i, **action)
                self.agent.feedback(result)
                add_dict(self.data, run_id, [{'context': context, 'action': action, 'result': result}])
                success = "Success!" if result["effect"]["Success"] else ""
                print(self.to_string(action, run_id, i, success))
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
    for i in range(1):
        runid = random.choice(range(10000))
        # Greedy agent
        # greedy_name = "greedy_runid_" + str(runid).zfill(4)
        # greedy_agent = GreedyAgent(greedy_name)
        # exp_greedy = Experiment(greedy_agent, greedy_name, run_idx=[runid])
        # exp_greedy.start()
        # Random agent
        # random_name = "random_runid_" + str(runid).zfill(4)
        # random_agent = RandomAgent(random_name)
        # exp_random = Experiment(random_agent, random_name, run_idx=[runid])
        # exp_random.start()
        # Multi beta agent
        # multib_name = "multibeta_runid_" + str(runid).zfill(4)
        # multib_agent = MultiBetaAgent(multib_name)
        # exp_multib = Experiment(multib_agent, multib_name, run_idx=[runid])
        # exp_multib.start()
        # Logistic agent
        # log_name = "log_runid_" + str(runid).zfill(4)
        # log_agent = NaiveLogisticAgent(log_name)
        # exp_log = Experiment(log_agent, log_name, run_idx=[runid])
        # exp_log.start()
        # Regularized regression agent
        # regreg_name = "regreg_runid_" + str(runid).zfill(4)
        # regreg_agent = RegRegressionAgent(regreg_name)
        # exp_regreg = Experiment(regreg_agent, regreg_name, run_idx=[runid])
        # exp_regreg.start()
        # Naive Bayesian agent
        # nb_name = "nb_runid_" + str(runid).zfill(4)
        # nb_agent = NaiveBayesAgent(nb_name)
        # exp_nb = Experiment(nb_agent, nb_name, run_idx=[runid])
        # exp_nb.start()
