from ipyparallel import Client, error
from mpi4py import MPI
import sys
import numpy as np

cluster = Client(profile="mpi")
comm = MPI.COMM_WORLD