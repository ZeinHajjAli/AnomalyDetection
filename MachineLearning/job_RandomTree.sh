#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --account=def-jrgreen
#SBATCH --mem-per-cpu=500G
#SBATCH --mail-user=zeinhajjali@cmail.carleton.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load java

java -Xmx498g -cp weka.jar weka.classifiers.trees.RandomTree -output-debug-info -no-cv -c last -K 0 -M 1.0 -V 0.001 -S 1 -depth 25 -t Data/Standardized19features.arff -split-percentage 66
