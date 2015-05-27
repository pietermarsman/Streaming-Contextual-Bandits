import numpy as np
from matplotlib import pyplot as plt

from misc import add_dict, create_key, create_directory


__author__ = 'pieter'

file = "1432738098.npy"
data = np.load("log/" + file).item()

stats = dict()

for runid, run_data in data.items():
    for record in run_data:
        success = record["result"]["effect"]["Success"]
        info = dict(record["context"]["context"], **record["action"])
        for k, v in info.items():
            create_key(stats, k, dict())
            create_key(stats[k], v, dict())
            stats[k][v]["count"] = stats[k][v].get("count", 0) + 1
            stats[k][v]["success"] = stats[k][v].get("success", 0) + success
            stats[k][v]["rate"] = stats[k][v]["success"] / stats[k][v]["count"]

create_directory("stats")
for stat_key in stats.keys():
    if stat_key != "ID":
        plt.clf()
        prop_keys = sorted(list(stats[stat_key].keys()))
        prop_values = [stats[stat_key][prop_key]["rate"] for prop_key in prop_keys]
        prop_counts = [stats[stat_key][prop_key]["count"] for prop_key in prop_keys]

        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        if isinstance(prop_keys[0], str) or isinstance(prop_keys[0], int):
            ind = np.arange(0, len(prop_keys), 1)
            ax0.bar(ind, prop_values, width=.8)
            ax1.plot(ind + .4, prop_counts)
            plt.xticks(ind + .4, tuple(prop_keys))
        if isinstance(prop_keys[0], float):
            ax0.scatter(prop_keys, prop_values)
            ax1.plot(prop_keys, prop_counts)
        plt.title("Runid's = " + str(list(data.keys())))
        ax0.set_ylim(0, 1)
        ax1.set_ylim(0, max(prop_counts) * 1.1)
        ax0.set_ylabel("Success rate")
        ax0.set_xlabel(stat_key)
        ax1.set_ylabel("Occurrences")
        plt.savefig("stats/" + stat_key)