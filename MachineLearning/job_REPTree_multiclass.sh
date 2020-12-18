#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --account=def-jrgreen
#SBATCH --mem-per-cpu=1498G
#SBATCH --mail-user=zeinhajjali@cmail.carleton.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=result_REPTree_multiclass.txt

module load java

java -Xmx1495g -cp weka.jar weka.classifiers.trees.REPTree -output-debug-info -no-cv -c last -M 2 -V 0.001 -N 3 -S 1 -t Data/FullDataSetWithAttackCat19Features.arff -split-percentage 66
