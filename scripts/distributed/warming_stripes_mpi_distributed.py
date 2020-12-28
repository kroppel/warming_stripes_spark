from mpi4py import MPI
import sys
import numpy as np
import math
import time

def get_data_states(path):
    data = list(map(lambda x: [x[i].strip() for i in [0] + np.arange(2,18).tolist()], map(lambda x: x.split(";"), open(path, "r").read().split("\n")[2:][:-1])))
    
    return data

def shuffle(data_dict, process_number, rank, key_ranges):
    return_dict = {}
    for foreign_rank in np.arange(0, process_number):
	temp1 = {}
	if foreign_rank != rank:
	    for key in key_ranges[foreign_rank]:
                temp1[key] = data_dict[key]
	    request = comm.isend(temp1, dest=foreign_rank)
	    request.wait()
	    temp2 = {}
            temp2.update(comm.irecv(source=i+1).wait())
	    for key in key_ranges[rank]:
	        data_dict[key] = data_dict[key] + temp2[key]

	      

def reducer(data_dict):
    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = np.mean(data_dict[key])
    
    return data_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#print(rank)

prog_start_time = time.time()

# Read in Data

data_path_hdfs = "data/data_oversized/"
data_path_local = "/home/vi46six/warming_stripes_spark/data/data_oversized/"
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
data = []
        
for month in months:
    data.append(list(get_data_states(data_path_local + "regional_averages_tm_" + month + ".txt")))

data_flattened = [pair for sublist in data for pair in sublist]

# data partitioning

key_ranges = {}
key_number = 2019-1881+1
process_number = 576
keys_per_process = math.ceil(key_number/process_number)

for rank in np.arange(0, process_number):
    if rank < (process_number-1):
        key_ranges[rank] = np.arange(1881+(rank*keys_per_process), 1881+((rank+1)*keys_per_process))
    else:
        key_ranges[rank] = np.arange(1881+(rank*keys_per_process), 2020)

#print(str(rank)+"; "+str(key_range))


data_partitioned = [[int(pair[0]), float(pair[1])] for pair in data_flattened if int(pair[0]) in key_range] 

# processing

data_dict = {}

for key in key_range:
    data_dict[key] = [pair[1] for pair in data_partitioned if pair[0]==key]

shuffle(data_dict, process_number, rank, key_ranges)

data_dict = reducer(data_dict)

# data aggregation on process 0

if rank != 0:
    request = comm.isend(data_dict, dest=0)
    request.wait()
else:
    for i in np.arange(process_number-1):
        data_dict.update(comm.irecv(source=i+1).wait())
    print(data_dict)
    
    prog_stop_time = time.time()
    exec_time = prog_stop_time-prog_start_time

    print("Execution time:"+str(exec_time))

	
