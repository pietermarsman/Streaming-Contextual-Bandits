import json
import os
import re

import numpy as np
from matplotlib import pyplot as plt

from misc import add_dict, create_key, create_directory, flatten, create_dictionary

__author__ = 'pieter'

# idea plot regret

STATS = ["count", "revenue", "rate"]
DIR = "agents/"


def create_1d_stats(run_data):
    """
    Create count and revenue stats per attribute value from a data dictionary
    :param run_data: [{context: context_dict, action: action_dict, result: result_dict}]
    :return: {context_key/action_key: {context_value/action_value: {count: int, revenue: int, rate: int}}}
    """
    stats = dict()
    means = dict()
    total = dict(count=0, revenue=0, rate=0)
    count = 0
    for record in run_data:
        success = record["result"]["effect"]["Success"]
        price = record["action"]["price"]
        info = dict(record["context"]["context"], **record["action"])
        for k, v in info.items():
            create_key(stats, k, dict())
            create_key(stats[k], v, dict())
            stats[k][v]["count"] = stats[k][v].get("count", 0) + 1
            stats[k][v]["revenue"] = stats[k][v].get("revenue", 0) + success * price
            stats[k][v]["rate"] = stats[k][v]["revenue"] / stats[k][v]["count"]
            total["count"] += stats[k][v]["count"]
            total["revenue"] += stats[k][v]["revenue"]
            total["rate"] += stats[k][v]["rate"]
            count += 1
    for stat in total.keys():
        means[stat] = total[stat] / count
    return stats, means


def create_2d_stats(run_data):
    """
    Create stats from a data dictionary
    :param run_data: [{context: context_dict, action: action_dict, result: result_dict}]
    """
    stats = dict()
    for record in run_data:
        success = record["result"]["effect"]["Success"]
        price = record["action"]["price"]
        for ck, cv in record["context"]["context"].items():
            for ak, av in record["action"].items():
                create_key(stats, ck, dict())
                create_key(stats[ck], ak, dict())
                create_key(stats[ck][ak], cv, dict())
                create_key(stats[ck][ak][cv], av, dict())
                stats[ck][ak][cv][av]["count"] = stats[ck][ak][cv][av].get("count", 0) + 1
                stats[ck][ak][cv][av]["revenue"] = stats[ck][ak][cv][av].get("revenue", 0) + success * price
                stats[ck][ak][cv][av]["rate"] = stats[ck][ak][cv][av]["revenue"] / stats[ck][ak][cv][av]["count"]
    return stats


def stat_dict_to_list(attr_stat_dict):
    """
    :param attr_stat_dict: {attribute_value: {count: int, revenue: int, rate: int}}
    :return: [attribute_value], {counts=[], revenue=[], rate=[]}
    """
    attr_values = sorted(list(attr_stat_dict.keys()))
    attr_stat_lists = dict()
    for attribute_value in attr_values:
        for stat_key in STATS:
            add_dict(attr_stat_lists, stat_key, [attr_stat_dict[attribute_value][stat_key]])
    return list(attr_values), attr_stat_lists


def plot_1d_stats(stats, means, name):
    for attr in stats.keys():
        if attr != "ID":
            attr_values, attr_stat_list = stat_dict_to_list(stats[attr])

            _, rate_axis = plt.subplots()
            count_axis = rate_axis.twinx()
            if isinstance(attr_values[0], str) or isinstance(attr_values[0], int):
                ind = np.arange(0, len(attr_values), 1)
                rate_axis.bar(ind, attr_stat_list[STATS[2]], width=.8)
                count_axis.plot(ind + .4, attr_stat_list[STATS[0]], 'k-')
                plt.xticks(ind + .4, tuple(attr_values))
            elif isinstance(attr_values[0], float):
                rate_axis.scatter(attr_values, attr_stat_list[STATS[2]])
                count_axis.plot(attr_values, attr_stat_list[STATS[0]], 'k-')
            else:
                raise TypeError("Unexpected datatype. Don't know how to plot")
            rate_axis.axhline(y=means["rate"])

            plt.title("Runid's = " + str(list(data.keys())))
            rate_axis.set_ylim(0, 20)
            rate_axis.set_ylabel("Avg revenue")
            rate_axis.set_xlabel(attr)
            count_axis.set_ylim(0, max(attr_stat_list[STATS[0]]) * 1.1)
            count_axis.set_ylabel("Occurrences")

            dir = "stats/" + name + "/rate/"
            create_directory(dir)
            plt.savefig(dir + attr)
            plt.close()


def plot_2d_stats(stats, name):
    for ck in stats:
        if ck != "ID":
            for ak in stats[ck]:
                context_values_keys = sorted(stats[ck][ak].keys())
                context_values = dict(zip(context_values_keys, range(len(context_values_keys))))
                action_values_keys = sorted(set(flatten(list(map(lambda x: x.keys(), stats[ck][ak].values())))))
                action_values = dict(zip(action_values_keys, range(len(action_values_keys))))
                ck_ak_stats = np.zeros((len(context_values), len(action_values)))
                maximum = 0
                for cv in sorted(stats[ck][ak]):
                    for av in sorted(stats[ck][ak][cv]):
                        ck_ak_stats[context_values[cv], action_values[av]] = stats[ck][ak][cv][av]["rate"]
                        maximum = max(maximum, stats[ck][ak][cv][av]["rate"])

                plt.imshow(ck_ak_stats.T, interpolation="none")
                plt.clim([0, maximum])
                if ck_ak_stats.shape[0] > ck_ak_stats.shape[1]:
                    plt.colorbar(orientation="horizontal")
                else:
                    plt.colorbar(orientation="vertical")

                plt.xticks(list(range(len(context_values))), list(context_values_keys), rotation='vertical')
                plt.yticks(list(range(len(action_values))), list(action_values_keys))
                plt.xlabel(ck)
                plt.ylabel(ak)
                plt.title("Revenue / show")

                dir = "stats/" + name + "/rate_interaction/"
                create_directory(dir)
                plt.savefig(dir + ck + "-" + ak)
                plt.close()


def plot_regret(run_data):
    max_reward = 15
    rewards = list(map(lambda record: record["result"]["effect"]["Success"] * record["action"]["price"], run_data))
    cum_rewards = np.cumsum(rewards)
    max_reward = np.cumsum(np.ones(cum_rewards.shape) * max_reward)
    plt.plot(max_reward - cum_rewards)
    filename = os.path.join("stats", name, "regret")
    plt.savefig(filename)
    plt.close()


def average_param_reward(files):
    average = {}
    for file in files:
        agent = np.load(os.path.join('agents', file)).item()
        log = np.load(os.path.join(DIR, file)).item()
        if 'reward' in log:
            if 'thomp(' in file:
                learnrate = agent['learnrate']
                regulizer = agent['regulizer']
                reward = log['reward']
                create_dictionary(average, learnrate, {})
                add_dict(average[learnrate], regulizer, [reward], [])
            elif 'greedy' in file:
                create_dictionary(average, 0.0, {})
                add_dict(average[0.0], 'greedy', [log['reward']], [])
    k2_length = 0
    lengths = {}
    for k1 in average:
        k2_length = max(k2_length, len(average[k1]))
        for k2 in average[k1]:
            create_dictionary(lengths, k1, {})
            lengths[k1][k2] = len(average[k1][k2])
            average[k1][k2] = sum(average[k1][k2]) / len(average[k1][k2])
    return average, lengths

files = [file for file in os.listdir(DIR) if "thomp(" in file or 'greedy' in file]
# for file in files:
#     name = file[:-4]
#     print("Processing: " + name)
#     create_directory("stats")
#     data = np.load(DIR + file).item()
#     stats_1d, means = create_1d_stats(list(data.values())[0])
#     stats_2d = create_2d_stats(list(data.values())[0])
#     plot_2d_stats(stats_2d, name)
#     plot_1d_stats(stats_1d, means, name)
#     plot_regret(list(data.values())[0])

averages, lengths = average_param_reward(files)
print(json.dumps(averages, sort_keys=True, indent=4, separators=(',', ': ')))
print(json.dumps(lengths, sort_keys=True, indent=4, separators=(',', ': ')))
