'''
This computes the gradients for a given combination of corruption method, gradient style and random seed (t-SNE initialization)
This performs the experiment on the local, class-based and global levels.

For each combination of level (local, class-based and global) and corruption method, gradient style and random seed,
this script returns a .csv file containing the results.
'''

#!/bin/bash

METHOD=$1
GRAD_STYLE=$2
SEED=$3

echo "Performing experiment with method="$METHOD" and attribution style="$GRAD_STYLE

# Hack
if [ "$METHOD" = "mean" ]; then
    echo "Changing mean to remove for global"
    METHOD_GLOBAL="remove"
else
    echo "No change in string needed"
    METHOD_GLOBAL=$METHOD
fi

source ./set_environment_variables_public.sh

FILE_NAME_CLASS='class_feats_method='$METHOD'_grad_style='$GRAD_STYLE
FILE_NAME_GLOBAL='global_feats_method='$METHOD_GLOBAL'_grad_style='$GRAD_STYLE
FILE_NAME_INDIVIDUAL='indv_feats_method='$METHOD'_grad_style='$GRAD_STYLE

mkdir $RESULTS_DIR/$FILE_NAME_CLASS
mkdir $RESULTS_DIR/$FILE_NAME_GLOBAL
mkdir $RESULTS_DIR/$FILE_NAME_INDIVIDUAL

RUN_ID=$(($SEED+30)) 
echo "Running global attribution experiments for seed "$RUN_ID

python src/attribution_experiment.py --tsne_output_dir $TSNE_OUTPUT_DIR_MNIST --data_dir $DATA_DIR_MNIST --data_id 0 --run_id $RUN_ID --method $METHOD_GLOBAL --step 250 --grad_style $GRAD_STYLE --indices_list 'top_pc' 'top_fs' 'top_ls_p_matrix' 'top_ls_q_matrix' 'top_attr_times_feat_unif' 'top_attr_unif' 'top_feat' --final_csv_path $RESULTS_DIR/$FILE_NAME_GLOBAL

wait

RUN_ID=$(($SEED+30)) 
echo "Running class-based attribution experiments for seed "$RUN_ID
python src/attribution_experiment_class_feats.py --tsne_output_dir $TSNE_OUTPUT_DIR_MNIST --data_dir $DATA_DIR_MNIST --data_id 0 --run_id $RUN_ID --method $METHOD --grad_style $GRAD_STYLE --step 250 --indices_list 'top_ls_class_p_matrix' 'top_ls_class_q_matrix' 'top_attr_times_feat_unif_class' 'top_attr_unif_class' 'top_feat_class'  --final_csv_path $RESULTS_DIR/$FILE_NAME_CLASS

wait

RUN_ID=$(($SEED+20)) 
echo "Running individual attribution experiments for seed "$RUN_ID
python src/attribution_experiment_individual_feats.py --tsne_output_dir $TSNE_OUTPUT_DIR_MNIST --data_dir $DATA_DIR_MNIST --data_id 0 --run_id $RUN_ID --method $METHOD --grad_style $GRAD_STYLE --step 250 --indices_list 'top_attr' 'attr_feat' 'attr_ge_0_feat_ge_0' 'top_attr_ge_0' 'feat_size' --final_csv_path $RESULTS_DIR/$FILE_NAME_INDIVIDUAL

wait
