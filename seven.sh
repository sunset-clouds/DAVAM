#!/bin/bash

# obtain directory paths
dir_curr=`pwd`
dir_temp="${dir_curr}-minimal"

# create a minimal code directory
sh ./scripts/create_minimal.sh ${dir_curr} ${dir_temp}
cd ${dir_temp}

# default arguments
nb_gpus=1
job_name="discrete-vae"

# parse arguments passed from the command line
extra_args=""
for i in "$@"
do
  case "$i" in
    -n=*|--nb_gpus=*)
    nb_gpus="${i#*=}"
    shift
    ;;
    -j=*|--job_name=*)
    job_name="${i#*=}"
    shift
    ;;
    *)
    # unknown option
    extra_args="${extra_args} ${i}"
    shift
    ;;
  esac
done
extra_args="${extra_args}"
echo ${extra_args} > extra_args
echo "Job name: ${job_name}"
echo "# of GPUs: ${nb_gpus}"
echo "extra arguments: ${extra_args}"

# update the configuration file
sed -i "s/nvidia.com\/gpu:\ 1/nvidia.com\/gpu:\ ${nb_gpus}/g" seven.yaml
sed -i "s/name:\ NB_GPUS/name:\ NB_GPUS\n\ \ \ \ value: ${nb_gpus}/g" seven.yaml

# start the seven job
seven create -conf seven.yaml -code `pwd` -name ${job_name}

# return to the main directory
cd ${dir_curr}
