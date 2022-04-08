#!/bin/bash


set -x
# source ./common.sh

# setting up base environment.
# following env variables are exported:
# - PS_HOSTS: the ps host in regard of PS-Worker training framework.
# - TRAINING_WORKERS: the training workers in both PS-Worker and RingAllReduce framework, comes in forms of host lists seperated by comma, like HOST1:HOST2...etc.
# - JOB_NAME: the job role (ps or worker) this host is assigned.
# - TASK_INDEX: the task index this host is assigned.
# - CPU_COMMON_MPI_PARAMETERS: basic cpu mpi common parameters for cases where mpi job with cpu workload is involved.
# - GPU_COMMON_MPI_PARAMETERS: basic gpu mpi common parameters for cases where mpi job with gpu workload is involved.
# predefined JAVA_HOME,CLASS_PATH, HADOOP_XXX environments are also exported.
# fyi, environment variables can be overrided later.
setup_env gpu

echo "we got ${TRAINING_WORKERS} for ring-based training"

# do your custom work here.
# for example, launch a data preprocessing and feeding service.
echo "no custom work to be done."


# run default mpi routine
# if you want to tune mpi, override CPU_COMMON_MPI_PARAMETERS or GPU_COMMON_MPI_PARAMETERS env variable or add custom options to it.
# for example,export GPU_COMMON_MPI_PARAMETERS="$GPU_COMMON_MPI_PARAMETERS -x NCCL_DEBUG=TRACE"
# if you don't need mpi, replace the following code with your training routine stub.
echo "run mpi job..."
myjob="sh train.sh"
run_mpi_job gpu "${myjob}"
res_code=$?
echo "res_code: $res_code \n"
if [ $res_code = 0 ] || [ $res_code = 1 ]
then
        echo "it's ok!\n"
        exit 0
else
        exit $res_code
fi