from mpi4py import MPI
import sys
import numpy as np
import math

def get_data(path):
    data = list(map(lambda x: [x[i].strip() for i in [0, -2]], map(lambda x: x.split(";"), open(path, "r").read().split("\n")[2:][:-1])))
    return data

def reducer(data_dict):
    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = np.mean(data_dict[key])
    return data_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

# Read in Data

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

# data partitioning

key_number = 2019-1881+1
process_number = 1
keys_per_process = math.ceil(key_number/process_number)
if rank < (process_number-1):
    key_range = np.arange(1881+(rank*keys_per_process), 1881+((rank+1)*keys_per_process))
else:
    key_range = np.arange(1881+(rank*keys_per_process), 2020)


data_partitioned = [[int(pair[0]), float(pair[1])] for pair in data_flattened if int(pair[0]) in key_range] 

# processing

data_dict = {}

for key in key_range:
    data_dict[key] = [pair[1] for pair in data_partitioned if pair[0]==key]

data_dict = reducer(data_dict)

# data aggregation on process 0

if rank != 0:
    request = comm.isend(data_dict, dest=0)
    request.wait()
else:
    for i in np.arange(process_number-1):
        data_dict.update(comm.irecv(source=i+1).wait())
    print(data_dict)
