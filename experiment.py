import threading
import time
from agents import RandomAgent
from communication import get_context, propose_page
import numpy as np
from misc import create_directory, add_dict

__author__ = 'pieter'


class Experiment(threading.Thread):

    MAX_I = 10000

    def __init__(self, agent, run_idx=[0]):
        super().__init__()
        self.agent = agent
        self.run_idx = run_idx
        self.data = dict()
        self.name = str(round(time.time()))

    def run(self):
        for run_id in self.run_idx:
            for i in range(self.MAX_I + 1):
                context = get_context(run_id, i)
                action = self.agent.decide(context)
                result = propose_page(run_id, i, **action)
                print("runid=" + str(run_id), "i=" + str(i), "Success!" if result["effect"]["Success"] else "")
                self.agent.feedback(result)
                add_dict(self.data, run_id, [{'context': context, 'action': action, 'result': result}])
            self.save()

    def save(self):
        create_directory("log")
        np.save('log/' + self.name, self.data)


if __name__ == "__main__":
    agent = RandomAgent()
    exp = Experiment(agent)
    exp.start()