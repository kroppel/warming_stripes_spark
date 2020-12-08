import numpy as np
import pyspark
from operator import add

# read data from given file
def get_data(path):
    data_de = np.loadtxt(fname=path, dtype=float, usecols=(0,18), skiprows=2, delimiter=";")
    return data_de

conf = pyspark.SparkConf()
sc = pyspark.SparkContext(master="local", appName="App Name", conf=conf)

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

result = rdd2.reduceByKey(lambda a, b: tuple(map(add, a, b))).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
temps = [pair[1] for pair in result]
print(temps)