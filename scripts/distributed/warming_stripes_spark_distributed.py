import numpy as np
import pyspark
from operator import add

# read average data from given file
def get_data_average(path):
    data = np.loadtxt(fname=path, dtype=float, usecols=(0,18), skiprows=2, delimiter=";")
    
    return data

def get_data_states(path):
    data = [[sublist[0],sublist[1:]] for sublist in np.loadtxt(fname=path, dtype=float, usecols=(0,) + tuple(np.arange(2,18).tolist()), skiprows=2, delimiter=";")]

    return data

# initialize spark and other variables
conf = pyspark.SparkConf()
sc = pyspark.SparkContext(master="local", appName="App Name", conf=conf)

data_path_local = "/home/vi46six/warming_stripes_spark/data/data_oversized/"
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
data = []

# read data        
"""
for month in months:
    data.append(list(get_data_average(data_path_local + "regional_averages_tm_" + month + "_oversized.txt")))

data_flattened = [pair for sublist in data for pair in sublist]

rdd2 = sc.parallelize(data_flattened).map(lambda x: tuple(x)).mapValues(lambda x: (x, 1))
"""


for month in months:
    data.append(list(get_data_states(data_path_local + "regional_averages_tm_" + month + "_oversized.txt")))

data_flattened = [pair for sublist in data for pair in sublist]

rdd2 = sc.parallelize(data_flattened).map(lambda x: tuple(x)).map(lambda x: (x[0], np.mean(x[1]))).mapValues(lambda x: (x, 1))

result = rdd2.reduceByKey(lambda a, b: tuple(map(add, a, b))).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
temps = [pair[1] for pair in result]
print(temps)

