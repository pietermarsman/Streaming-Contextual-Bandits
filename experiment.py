import os
import random
import threading
import time
import datetime

import numpy as np

from agents import ThompsonLogisticAgent
from agents import GreedyAgent
from agents import MultiBetaAgent
from agents import RandomAgent
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


enabled = {"greedy": True, "random": False, "multib": False, "thomp": False}
# learnrates = [0.01] #[0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
# regulizers = [1e-3] #[0.01, 0.005, 0.001, 0.0005, 0.0001]
n_exp = 1
# priors = ThompsonLogisticAgent.parse_priors([os.path.join('agents', file) for file in os.listdir('agents') if 'thomp(0.0100,0.0010)' in file])

if __name__ == "__main__":
    now = time.time()
    experiments = []
    for runid in range(10001, 10011):
        # runid = random.choice(range(10000))
        str_runid = str(runid).zfill(4)
        # Greedy agent
        if enabled["greedy"]:
            greedy_name = "greedy_runid_" + str(runid).zfill(4)
            greedy_agent = GreedyAgent(greedy_name)
            exp_greedy = Experiment(greedy_agent, greedy_name, run_idx=[runid])
            experiments.append(exp_greedy)
            exp_greedy.start()
        # Random agent
        # if enabled["random"]:
        #     random_name = "random_runid_" + str(runid).zfill(4)
        #     random_agent = RandomAgent(random_name)
        #     exp_random = Experiment(random_agent, random_name, run_idx=[runid])
        #     exp_random.start()
        # Multi beta agent
        # if enabled["multib"]:
        #     multib_name = "multibeta_runid_" + str(runid).zfill(4)
        #     multib_agent = MultiBetaAgent(multib_name)
        #     exp_multib = Experiment(multib_agent, multib_name, run_idx=[runid])
        #     experiments.append(exp_multib)
        #     exp_multib.start()
        # Bootstrap Thompson sampling poor man's Bayes streaming logistic regression
        # if enabled["thomp"]:
        #     for learnrate in learnrates:
        #         for regulizer in regulizers:
        #             thomp_name = "thomp(%.4f,%.4f)_runid_%s" % (learnrate, regulizer, str_runid)
        #             thomp_agent = ThompsonLogisticAgent(thomp_name, learnrate, regulizer, 200, 100, prior=priors)
        #             exp_thomp = Experiment(thomp_agent, thomp_name, run_idx=[runid])
        #             experiments.append(exp_thomp)
        #             exp_thomp.start()
    while any(map(lambda x: x.is_alive(), experiments)):
        time.sleep(10)
        # experiments[1].agent.plot(include=["price"], exclude=['ID', 'Agent'])
        # thomp_sum = [exp.data.get('cum_reward', 0) for exp in experiments if 'thomp' in exp.name]
        greedy_sum = [exp.data.get('cum_reward', 0) for exp in experiments if 'greedy' in exp.name]
        # print('thomp', sum(thomp_sum) / max(len(thomp_sum), 1))
        print('greedy', sum(greedy_sum) / max(len(greedy_sum), 1))
    for experiment in experiments:
        print(experiment.data["reward"], experiment.name)
    print("Duration:", time.time() - now)


