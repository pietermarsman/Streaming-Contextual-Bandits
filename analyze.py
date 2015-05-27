import numpy as np
from matplotlib import pyplot as plt

__author__ = 'pieter'

def add_dict(dict, key, value):
    if key in dict:
        dict[key] += value
    else:
        dict[key] = value

file = "1432728005.npy"
data = np.load("log/" + file)

successes = dict()
counts = dict()

for record in data:
    success = record["result"]["effect"]["Success"]
    for context_key, context_value in record["context"]["context"].items():
        key = context_key + "=" + str(context_value)
        add_dict(counts, key, 1)
        add_dict(successes, key, success)
    for action_key, action_value in record["action"].items():
        key = action_key + "=" + str(action_value)
        add_dict(counts, key, 1)
        add_dict(successes, key, success)

age = []
age_rate = []
for key in sorted(counts.keys()):
    print(key, ":", successes[key], "/", counts[key], "=", successes[key] / counts[key])
    if key.startswith("Age="):
        age.append(float(key[4:]))
        age_rate.append(successes[key] / counts[key])
plt.scatter(age, age_rate)
plt.xlabel("Age")
plt.ylabel("Success rate")
plt.show()