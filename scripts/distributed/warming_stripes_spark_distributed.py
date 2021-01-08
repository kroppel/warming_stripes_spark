import numpy as np
import pyspark
from operator import add
import time

# read average data from given file

def get_data_states(path, local=True):
    if not local:
        data = (sc.textFile(path)
		.filter(lambda x: True if x[0] in ["1", "2"] else False)
		.map(lambda x: x.split(";"))
		.map(lambda x: [x[0], x[2:18]])
		.map(lambda x: [int(x[0]), list(map(float, map(str.strip, map(str,x[1]))))]))
    else:
        data = [[sublist[0],sublist[1:]] for sublist in np.loadtxt(fname=path, dtype=float, usecols=(0,) + tuple(np.arange(2,18).tolist()), skiprows=2, delimiter=";")]

    return data

# start time measurement
start = time.time()

# initialize spark and other variables
conf = (pyspark.SparkConf()
	.set("spark.driver.cores", "36")
	.set("spark.executor.cores", "36")
	.set("spark.driver.memory", "64g")
	.set("spark.executor.memory", "64g")
	.set("spark.yarn.executor.memoryOverhead", "4096")
	.set("spark.driver.memoryOverhead", "32g") 
	.set("spark.executor.memoryOverhead", "32g") 
	.set("spark.default.parallelism", 30000)
        .set("yarn.nodemanager.resource.memory-mb", "196608")
	.set("spark.dynamicAllocation.enabled", "True")
	.set("spark.dynamicAllocation.minExecutors", "15")
	.set("spark.dynamicAllocation.maxExecutors", "15")
	)
sc = pyspark.SparkContext(appName="Warming Stripes Spark", conf=conf)

data_path_hdfs = "data/data_oversized_v3/"
data_path_local = "/home/vi46six/warming_stripes_spark/data/data_oversized_v3/"
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
data = []
use_local_fs = True

# read data        
if use_local_fs:
    for month in months:
        data.append(list(get_data_states(data_path_local + "regional_averages_tm_" + month + ".txt", use_local_fs)))

    data_flattened = [pair for sublist in data for pair in sublist]

    intermediate = time.time()    

    rdd2 = (sc.parallelize(data_flattened)
	    .map(lambda x: tuple(x))
	    .map(lambda x: (x[0], np.mean(x[1])))
	    .mapValues(lambda x: (x, 1)))

else:
    for month in months:
        if month == "01":
	    rdd2 = get_data_states(data_path_hdfs + "regional_averages_tm_" + month + ".txt", use_local_fs)
        else:
	    rdd2.union(get_data_states(data_path_hdfs + "regional_averages_tm_" + month + ".txt", use_local_fs))

    intermediate = time.time()

    rdd2 = (rdd2.map(lambda x: tuple(x))
	    .map(lambda x: (x[0], np.mean(x[1])))
	    .mapValues(lambda x: (x, 1)))


result = rdd2.reduceByKey(lambda a, b: tuple(map(add, a, b))).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
#temps = [pair[1] for pair in result]

# stop time measurement
stop = time.time()

result.sort(key=lambda x: x[0])

file = open("/home/vi46six/warming_stripes_spark/out/spark/outfile.out", "w")
file.write(str(result))
file.write("\nExecution time: "+str(stop-start)+"\nData Loading: "+str(intermediate-start)+"\nPure execution: "+str(stop-intermediate))
file.close()

