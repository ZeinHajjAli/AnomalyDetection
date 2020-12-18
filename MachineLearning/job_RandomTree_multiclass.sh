#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --account=def-jrgreen
#SBATCH --mem-per-cpu=498G
#SBATCH --mail-user=zeinhajjali@cmail.carleton.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=result_RandomTree_multiclass.txt

module load java

java -Xmx496g -cp weka.jar weka.classifiers.trees.RandomTree -output-debug-info -no-cv -c last -K 0 -M 1.0 -V 0.001 -S 1 -depth 100 -t Data/FullDataSetWithAttackCat19Features.arff -split-percentage 66
