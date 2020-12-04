import numpy as np
import pyspark
from operator import add

# read data from given file
def get_data(path):
    data_de = np.loadtxt(fname=path, dtype=float, usecols=(0,18), skiprows=2, delimiter=";")
    return data_de

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
temps = list(result_local.values())
print(temps)