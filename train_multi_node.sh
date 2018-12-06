#!/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-16"
#PJM -L "elapse=4:00:00"
#PJM -L "vnode=4"
#PJM -L "vnode-core=36"
#PJM -j
#PJM -X
#
#######################################

# load and check module.
module load cuda/8.0 hdf5/1.10.1-threads
module load openmpi/1.10.7
module unload intel/2017

mpirun -n 4 -np 16 --map-by ppr:4:node --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} \
    python train_multi_node.py -b 64 -e 200 -o cnn3_multi_node --communicator 'non_cuda_aware'

module load intel/2017
module unload openmpi/1.10.7
module unload cuda/8.0 hdf5/1.10.1-threads
