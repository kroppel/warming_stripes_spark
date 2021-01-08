from mpi4py import MPI
import sys
import numpy as np
import math
import time

def get_data_states(path):
    data = list(map(lambda x: [x[i].strip() for i in [0] + np.arange(2,18).tolist()], map(lambda x: x.split(";"), open(path, "r").read().split("\n")[2:][:-1])))
    return data
     
def reducer(data_dict):
    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = np.mean(data_dict[key])
    
    return data_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

prog_start_time = time.time()

# read in data completely

data_path_beegfs = "/beegfs/vi46six/data/data_oversized_v3/"
data_path_local = "/home/vi46six/warming_stripes_spark/data/data_oversized_v3/"
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
data = []
        
for month in months:
    data.append(list(get_data_states(data_path_beegfs + "regional_averages_tm_" + month + ".txt")))

data_flattened = [pair for sublist in data for pair in sublist]

# intermediate time measurement
prog_intermediate_time = time.time()

# data partitioning locally

key_ranges = {}
key_number = 2019-1881+1
# number of total processes
process_number = 60
keys_per_process = math.ceil(key_number/process_number)
# id of process that gets last partition assigned
last_process = math.ceil(key_number/keys_per_process) - 1


for r in np.arange(0, process_number):
    if r < (last_process):
        key_ranges[r] = np.arange(1881+(r*keys_per_process), 1881+((r+1)*keys_per_process))
    elif r == (last_process):
        key_ranges[r] = np.arange(1881+(r*keys_per_process), 2020)
    else:
        key_ranges[r] = np.asarray([])

data_partitioned = [[int(pair[0]), float(pair[1])] for pair in data_flattened if int(pair[0]) in key_ranges.get(rank, [])] 

# processing

data_dict = {}

for key in key_ranges[rank]:
    data_dict[key] = [pair[1] for pair in data_partitioned if pair[0]==key]

data_dict = reducer(data_dict)

# data aggregation on process 0

if rank != 0 and rank <= last_process:
    request = comm.isend(data_dict, dest=0)
    request.wait()
elif rank == 0:
    for i in np.arange(last_process):
        data_dict.update(comm.irecv(source=i+1).wait())
    print(data_dict)
    
    prog_stop_time = time.time()
    exec_time = prog_stop_time-prog_start_time

    print("Execution time:"+str(exec_time))
    print("Data loading: "+str(prog_intermediate_time-prog_start_time))
    print("Pure execution: "+str(prog_stop_time-prog_intermediate_time))
