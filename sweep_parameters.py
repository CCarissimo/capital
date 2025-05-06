import numpy as np
from itertools import product

pow2 = np.array([2**i for i in range(1, 11)])
print(pow2, len(pow2))

n_processes = []

for i, na in enumerate(pow2):
    half = na/2
    exponent = i + 1
    nproc = np.array([2**i for i in range(0, exponent)])
    n_processes.append(nproc)

print(n_processes)

params = []
nparams = 0

for i, na in enumerate(pow2):
    combined = [(na, nproc) for nproc in n_processes[i]]
    params.append(combined)
    nparams += len(combined)

print(params)
print(len(params))
print(nparams)