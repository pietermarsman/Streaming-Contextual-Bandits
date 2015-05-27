import os
import numpy as np
from matplotlib import pyplot as plt

from misc import add_dict, create_key, create_directory


__author__ = 'pieter'


STATS = ["count", "success", "rate"]
DIR = "log/"

def create_stats(run_data):
    """
    Create stats from a data dictionary
    :param run_data: [{context: context_dict, action: action_dict, result: result_dict}]
    :return: {context_key/action_key: {context_value/action_value: {count: int, success: int, rate: int}}}
    """
    stats = dict()
    for record in run_data:
        success = record["result"]["effect"]["Success"]
        info = dict(record["context"]["context"], **record["action"])
        for k, v in info.items():
            create_key(stats, k, dict())
            create_key(stats[k], v, dict())
            stats[k][v]["count"] = stats[k][v].get("count", 0) + 1
            stats[k][v]["success"] = stats[k][v].get("success", 0) + success
            stats[k][v]["rate"] = stats[k][v]["success"] / stats[k][v]["count"]
    return stats


def stat_dict_to_list(attr_stat_dict):
    """
    :param attr_stat_dict: {attribute_value: {count: int, success: int, rate: int}}
    :return: [attribute_value], {counts=[], success=[], rate=[]}
    """
    attr_values = attr_stat_dict.keys()
    attr_stat_lists = dict()
    for attribute_value in attr_values:
        for stat_key in STATS:
            add_dict(attr_stat_lists, stat_key, [attr_stat_dict[attribute_value][stat_key]])
    return list(attr_values), attr_stat_lists


def plot_rate_stats(stats, runid):
    for attr in stats.keys():
        if attr != "ID":
            attr_values, attr_stat_list = stat_dict_to_list(stats[attr])

            _, rate_axis = plt.subplots()
            count_axis = rate_axis.twinx()
            if isinstance(attr_values[0], str) or isinstance(attr_values[0], int):
                ind = np.arange(0, len(attr_values), 1)
                rate_axis.bar(ind, attr_stat_list[STATS[2]], width=.8)
                count_axis.plot(ind + .4, attr_stat_list[STATS[0]])
                plt.xticks(ind + .4, tuple(attr_values))
            if isinstance(attr_values[0], float):
                rate_axis.scatter(attr_values, attr_stat_list[STATS[2]])
                count_axis.plot(attr_values, attr_stat_list[STATS[0]])

            plt.title("Runid's = " + str(list(data.keys())))
            rate_axis.set_ylim(0, 1)
            rate_axis.set_ylabel("Success rate")
            rate_axis.set_xlabel(attr)
            count_axis.set_ylim(0, max(attr_stat_list[STATS[0]]) * 1.1)
            count_axis.set_ylabel("Occurrences")

            dir = "stats/runid_" + str(runid).zfill(4) + "/"
            create_directory(dir)
            plt.savefig(dir + attr)
            plt.close()


for file in os.listdir(DIR):
    create_directory("stats")
    data = np.load(DIR + file).item()
    for runid, run_data in data.items():
        stats = create_stats(run_data)
        plot_rate_stats(stats, runid)
