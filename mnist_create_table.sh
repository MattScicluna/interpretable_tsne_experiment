"""
This merges all the tables produced from mnist_experiment.sh
You will end up with a .csv file for each combination of level (local, class-based and global) and corruption method and gradient style.
This file will contain all the results for all the random seeds. This file can be looked at in the Jupyter notebooks.
"""
#!/bin/bash

source ./set_environment_variables_public.sh

# Activate virtual env
module load python/3.8
source $ENV_LOC/bin/activate
module load python/3.8  # fixes problem with matplotlib


python src/combine_mnist_tables.py --result_dir $RESULTS_DIR --final_csv_path $RESULTS_DIR --method "permute" --grad_style "grad_norm"
python src/combine_mnist_tables.py --result_dir $RESULTS_DIR --final_csv_path $RESULTS_DIR --method "mean" --grad_style "grad_norm"
python src/combine_mnist_tables.py --result_dir $RESULTS_DIR --final_csv_path $RESULTS_DIR --method "permute" --grad_style "kl_obj"
python src/combine_mnist_tables.py --result_dir $RESULTS_DIR --final_csv_path $RESULTS_DIR --method "mean" --grad_style "kl_obj"
