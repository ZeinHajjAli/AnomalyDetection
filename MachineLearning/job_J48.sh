#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --account=def-jrgreen
#SBATCH --mem-per-cpu=507G
#SBATCH --mail-user=zeinhajjali@cmail.carleton.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load java

java -Xmx506g -cp weka.jar weka.classifiers.trees.J48 -output-debug-info -no-cv -c last -C 0.25 -M 2 -N 3 -Q 1 -t Data/Standardized19features.arff -split-percentage 66
