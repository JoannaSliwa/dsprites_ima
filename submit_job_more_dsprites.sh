#!/bin/bash

EXPROOT="/cobra/u/jsliwa/Documents/NuX"
JOBTIME="1-00:00:00"
NUMJOBS=1
GPU="None"

while getopts ":c:t:j:g:" o; do
    case "${o}" in
	t)
	    JOBTIME=${OPTARG}
	    ;;
	g)
	    GPU=${OPTARG}
            ;;
    esac
done

if [ "$GPU" = "None" ]
then
    sbatch --job-name="dsprites" --time=${JOBTIME} --nodes=1 --ntasks-per-node=1 --cpus-per-task=20 --mem=40000 job_dsprites.sbatch
else
    GRES="gpu:${GPU}:1"
    sbatch --job-name="dsprites" --time=${JOBTIME} --constraint="gpu" --nodes=1 --ntasks-per-node=1 --gres=${GRES} --cpus-per-task=20 --mem=90000 job_dsprites.sbatch
fi

