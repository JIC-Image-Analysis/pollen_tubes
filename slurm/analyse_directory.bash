#!/bin/bash -e
#SBATCH -p rg-mh # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of tasks
#SBATCH --mem 1000 # memory pool for all cores
#SBATCH -t 0-2:00 # time (D-HH:MM)
#SBATCH -D /usr/users/JIC_a5/olssont/projects/pollen-tube
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=tjelvar.olsson@jic.ac.uk # send-to address

SCRIPT="scripts/nikonE800_annotate.py"
INPUT_DIR=${1}
INPUT_DIR=${INPUT_DIR%/}  # Remove trailing slash.
OUTPUT_DIR="$INPUT_DIR/image_analysis"
PYTHON="/nbi/software/testing/linuxbrew/default/src/bin/python"

srun $PYTHON $SCRIPT "$INPUT_DIR" "$OUTPUT_DIR"

