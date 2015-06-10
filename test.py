import math
import numpy as np

__author__ = 'pieter'

real_mean = 15
real_var = 10
ass = np.random.randn(1000) * real_var + real_mean

actual_mean = np.mean(ass)
actual_std = np.std(ass)
actual_diff = np.mean(ass - ass.mean())

print("Real mean: {}".format(real_mean))
print("Actual mean: {}".format(actual_mean))
print("Real std: {}".format(real_var))
print("Actual std: {}".format(actual_std))

computed_mean = ass[0]
computed_var = 0
computed_diff = 0
count = 1

for a in ass:
    temp_mean = computed_mean
    computed_mean += (a - computed_mean) / count
    computed_var += (a - temp_mean) * (a - computed_mean)
    computed_diff += a - temp_mean
    count += 1
computed_std = math.sqrt(computed_var / (count - 1))
print("Computed mean: {}".format(computed_mean))
print("Computed std: {}".format(computed_std))