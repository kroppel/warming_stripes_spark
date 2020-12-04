import numpy as np
#import matplotlib.pyplot as plt
import pyspark
from operator import add

# read data from given file
def get_data(path):
    data_de = np.loadtxt(fname=path, dtype=float, usecols=(0,18), skiprows=2, delimiter=";")
    return data_de

# visualize warming stripes
"""def show_warming_stripes(data):
    temps = data
    stacked_temps = np.stack((temps, temps))

    plt.figure(figsize=(4,18))
    img = plt.imshow(stacked_temps, cmap='RdBu_r', aspect=40, )

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("stripes.png", bbox_inches = 'tight', pad_inches = 0, dpi=400)
"""

sc = pyspark.SparkContext("local", "App Name")

months = []
data = []

for i in np.arange(start=1, stop=13):
    if i < 10:
        months.append("0"+str(i))
    else:
        months.append(str(i))
        
for month in months:
    data.append(list(get_data("../data/regional_averages_tm_"+month+".txt")))

data_flattened = [pair for sublist in data for pair in sublist]

rdd2 = sc.parallelize(data_flattened).map(lambda x: tuple(x)).mapValues(lambda x: (x, 1))

result_local = rdd2.reduceByKeyLocally(lambda a, b: tuple(map(add, a, b)))

for key in result_local.keys():
    result_local[key] = result_local[key][0]/result_local[key][1]
temps1 = list(result_local.values())
print(temps1)
#show_warming_stripes(temps1)

"""
result = rdd2.reduceByKey(lambda a, b: tuple(map(add, a, b))).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
temps2 = [pair[1] for pair in result]
show_warming_stripes(temps2)
"""