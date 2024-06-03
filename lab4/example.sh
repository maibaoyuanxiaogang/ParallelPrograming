#mpi.sh
#!/bin/sh
#PBS -N test 
#PBS -l nodes=master_vir9

pssh -h $PBS_NODEFILE mkdir -p /home/s2212895/mpi 1>&2
scp master:/home/s2212895/mpi/test /home/s2212895/mpi
pscp -h $PBS_NODEFILE /home/s2212895/mpi/test /home/s2212895/mpi 1>&2
mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2212895/mpi/test