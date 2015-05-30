import random
import threading
import time

import numpy as np

from agents import GreedyAgent, RandomAgent
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
        np.save('log/' + self.name, self.data)
        np.save('agents/' + self.agent.name, self.agent.to_saveable())

    def to_string(self, action, run_id, i, success):
        return "runid={}, i={}, reward={:.4f}, action={} {}".format(run_id, i, self.agent.cum_reward / (i + 1), action,
                                                                     success)


if __name__ == "__main__":
    for i in range(3):
        runid = random.choice(range(10000))
        greedy_name = "greedy_runid_" + str(runid).zfill(4)
        random_name = "random_runid_" + str(runid).zfill(4)
        greedy_agent = GreedyAgent(greedy_name)
        random_agent = RandomAgent(random_name)
        exp_greedy = Experiment(greedy_agent, greedy_name, run_idx=[runid])
        exp_random = Experiment(random_agent, random_name, run_idx=[runid])
        exp_greedy.start()
        exp_random.start()