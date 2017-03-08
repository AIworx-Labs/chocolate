#!/usr/bin/env bash
# Total number of steps
steps=50
# Number of parallel jobs to execute
parallel=4

# Change to main script directory
cd ../simple

# Execute all tasks
for s in {1..${steps}..${parallel}}; do
    for p in {1..${parallel}}; do
        python sklearn-gbt.py &
    done
    # Wait for parallel jobs to finish
    wait
done