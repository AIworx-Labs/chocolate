#!/usr/bin/env bash
# Some flags have been omitted for demonstration purpose
# Refer to your scheduler documentation for necessary flags
#PBS -N ChocolatyJob
#PBS -l nodes=1:ppn=8
#PBS -l walltime=5:00:00
#PBS -t [1-50]%4

# Change to application directory
cd chocolate/examples/simple

# Start a single job per node
python sklearn-gbt.py

# Optionally, you can start multiple jobs per node (use the multiple cores)
# Comment out the last 'python sklearn-gbt.py' and uncomment the following 5 lines
#python sklearn-gbt.py &
#python sklearn-gbt.py &
#python sklearn-gbt.py &
#python sklearn-gbt.py &
#wait
