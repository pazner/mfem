#!/usr/bin/env bash
#SBATCH --job-name=duffy_run_all # Give the job a name.
#SBATCH --partition=normal       # Use the normal partition.
#SBATCH --time=24:00:00          # 24-hour time-limit (format HH:MM:SS).
#SBATCH --ntasks=64              # 64 CPU cores
#SBATCH --nodes=1                # 1 node only
#SBATCH --mem=0                  # Full node memory
#SBATCH --error=duffy_results.err.txt
#SBATCH --output=duffy_results.out.txt

printf "Starting the run…\n\n"

julia --project run_cases.jl --full
